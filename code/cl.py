import pickle
import random
import numpy as np
from models import BasicModel
import torch
'''
参数共享: 所有用户和商品的嵌入
for _ in epochs:
    根据pair的相似性划分一次, 内部迭代若干次(默认50次)
    共享的用户和商品嵌入在每个sub-model中传播一次, 得到最后一层的输出
    根据sub-models的最后输出, 计算loss, 优化模型参数, 用户和商品嵌入是唯一参数
根据最后的用户和商品嵌入, 最后划分一次
'''


def cl_train(dataset_path, train_dict, num_shards, max_iters, epochs):
    with open(dataset_path + '/user_pretrain.pk', 'rb') as f:
        init_users_embeddings: np.ndarray = pickle.load(f, encoding='latin1')
    with open(dataset_path + '/item_pretrain.pk', 'rb') as f:
        init_items_embeddings: np.ndarray = pickle.load(f, encoding='latin1')
    # 元素为字典的列表, 每个字典代表一个分块中的交互, 字典的key是用户, value是用户的item集合
    for _ in range(epochs):
        sub_train_dicts: list[dict] = interaction_based_partition(init_users_embeddings, init_items_embeddings, train_dict, num_shards, max_iters)
        final_shards_embeddings = propagation(init_users_embeddings, init_items_embeddings, sub_train_dicts, num_shards)
        cl_loss = CLLoss(final_shards_embeddings, init_users_embeddings, init_items_embeddings)
        cl_loss.step()
    sub_train_dicts = interaction_based_partition(init_users_embeddings, init_items_embeddings, train_dict, num_shards, max_iters)


def interaction_based_partition(uidW, iidW, train_dict, num_shards, max_iters):
    # get_data_interactions_1
    data = []
    for i in train_dict:
        for j in train_dict[i]:
            data.append([i, j])
    # Randomly select k centroids
    max_data = 1.2 * len(data) // num_shards
    centroids = random.sample(data, num_shards)
    # centro emb
    centroembs = []
    for i in range(num_shards):
        temp_u = uidW[centroids[i][0]]
        temp_i = iidW[centroids[i][1]]
        centroembs.append([temp_u, temp_i])

    for _ in range(max_iters):
        C = [{} for i in range(num_shards)]
        C_num=[0 for i in range(num_shards)]
        Scores = {}
        for i in range(len(data)):
            for j in range(num_shards):
                score_u = eucli_dist2(uidW[data[i][0]],centroembs[j][0])
                score_i = eucli_dist2(iidW[data[i][1]],centroembs[j][1])
                Scores[i, j] = -score_u * score_i
        Scores = sorted(Scores.items(), key=lambda x: x[1], reverse=True)
        fl = set()
        for i in range(len(Scores)):
            # Scores[i][0][0] == pair_id
            if Scores[i][0][0] not in fl:
                if C_num[Scores[i][0][1]] < max_data:
                    # data[Scores[i][0][0]][0] == user_id
                    if data[Scores[i][0][0]][0] not in C[Scores[i][0][1]]:
                        C[Scores[i][0][1]][data[Scores[i][0][0]][0]]=[data[Scores[i][0][0]][1]]
                    else:
                        C[Scores[i][0][1]][data[Scores[i][0][0]][0]].append(data[Scores[i][0][0]][1])
                    fl.add(Scores[i][0][0])
                    C_num[Scores[i][0][1]] +=1
        centroembs_next = []
        for i in range(num_shards):
            temp_u = []
            temp_i = []
            for j in C[i].keys():
                for l in C[i][j]:
                    temp_u.append(uidW[j])
                    temp_i.append(iidW[l])
            centroembs_next.append([np.mean(temp_u), np.mean(temp_i)])

        loss = 0.0
        for i in range(num_shards):
            score_u = eucli_dist2(centroembs_next[i][0], centroembs[i][0])
            score_i = eucli_dist2(centroembs_next[i][1], centroembs[i][1])
            loss += (score_u * score_i)
        centroembs = centroembs_next
        # for i in range(dataloader.num_shards):
        #     print(C_num[i])
        print(C_num)
        print(_, loss)
    # users = [[] for i in range(dataloader.num_shards)]
    # items = [[] for i in range(dataloader.num_shards)]
    # for i in range(dataloader.num_shards):
    #     # dict.keys()或者dict.values()返回的是dict_keys对象，dict_items()返回的是dict_items对象
    #     # dict_keys对象和dict_items对象都是可迭代的，更高效，但是不能序列化，pickle会出错
    #     # Python低版本中不是返回dict_keys对象或者dict_items对象，不存在问题
    #     users[i] = list(C[i].keys())  
    #     for j in C[i].keys():
    #         for l in C[i][j]:
    #             if l not in items[i]:
    #                 items[i].append(l)

    return C  # , users, items

def eucli_dist2(x, y):
    return np.sum(np.square(x - y))