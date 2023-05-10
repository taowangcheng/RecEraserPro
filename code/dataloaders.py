import torch
import numpy as np
import random
import scipy.sparse as sp
import pickle
from time import time
from typing import Union

from configuration import Config
import display


class BasicDataLoader():
    def __init__(self, config: Config):
        # 邻接矩阵划分配置
        self.a_hat_split = config.a_hat_split
        self.a_hat_folds = config.a_hat_folds
        # 数据集配置
        self.dataset_path = config.dataset_path
        self.train_file = self.dataset_path + '/train.txt'
        self.test_file = self.dataset_path + '/test.txt'
        # 抽样配置
        self.train_batch_size = config.train_batch_size
        self.test_batch_size = config.test_batch_size
        # 设备配置
        self.device = config.device
        # 加载数据集
        self.n_users: int
        self.m_items: int
        self.alive_users: list[int]
        self.train_dict: dict
        self.valid_dict: dict
        self.test_dict: dict
        self.train_size: int
        self.norm_adj: Union[torch.Tensor, list[torch.Tensor]]
        self.sub_dataloaders: None or list[BasicDataLoader]
        self.dom_dataloader: Union[None, SpilitDataLoader] = None

    
    def sample1(self):
        """
        the original impliment of BPR Sampling in LightGCN
        从all_users中随机抽取train_size个用户, 每个用户选取1个正样本和1个负样本
        可能有用户是孤立点, 没有正样本, 会跳过
        在数据稀疏加剧的情况下, 可能会导致有很多孤立点, 返回的三元组数量会明细小于train_size?
        :return:
            np.array
        """
        user_num = self.train_size
        users = np.random.randint(0, self.n_users, user_num)  # 直接抽train_size个用户?
        S = []
        for user in users:
            pos_items = self.train_dict.get(user)  # user在训练集中可能没有商品，在字典中没有对应的key
            # posForUser = allPos[user]
            if pos_items is None:
                continue
            indexes = np.random.randint(0, len(pos_items))
            pos_item = pos_items[indexes]
            while True:
                neg_item = np.random.randint(0, self.m_items)
                if neg_item in pos_items:
                    continue
                else:
                    break
            S.append([user, pos_item, neg_item])
        return np.array(S)

    def sample2(self, neg_ratio = 1):
        """
        从alive_users中选取train_batch_size个用户, 每个用户选取1个正样本和neg_ratio(default=1)个负样本
        """
        # if self.train_batch_size <= self.n_users:
        if self.train_batch_size <= len(self.alive_users):
            users = random.sample(self.alive_users, self.train_batch_size)
        else:
            users = [random.choice(self.alive_users) for _ in range(self.train_batch_size)]
        def sample_pos_items_for_u(u, num):
            pos_items = self.train_dict[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]
                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch
        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.m_items, size=1)[0]
                if neg_id not in self.train_dict[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items
        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
        return users, pos_items, neg_items    
    
    def test2valid_test(self):
        interactions = []
        for user, items in self.test_dict.items():
            for item in items:
                interactions.append([user, item])
        num_interactions = len(interactions)
        random.shuffle(interactions)
        valid_interactions = interactions[:num_interactions//2]
        test_interactions = interactions[num_interactions//2:]
        valid_dict = {}
        test_dict = {}
        for user, item in valid_interactions:
            valid_dict.setdefault(user, [])
            valid_dict[user].append(item)
        for user, item in test_interactions:
            test_dict.setdefault(user, [])
            test_dict[user].append(item)
        self.valid_dict = valid_dict
        self.test_dict = test_dict

class NormalDataLoader(BasicDataLoader):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """
    def __init__(self, config: Config):
        super(NormalDataLoader, self).__init__(config)
        display.color_print(f'loading [{config.dataset_path}]')
        # 数据集生成
        self.n_users = 0
        self.m_items = 0
        self.train_size = 0  # 交互数
        self.test_size = 0  # 交互数
        self.alive_users = []  # 有交互的用户
        self.train_dict, self.test_dict = {}, {}  # dict格式, {user: [item1, item2, ...]}
        # train_unique_users, test_unique_users = [], []
        # train_items, train_users = [], []  # COO格式
        # test_items, test_users = [], []  # COO格式
        with open(self.train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    uid = int(l[0])
                    items = [int(i) for i in l[1:]]
                    items.sort() # 商品不一定有序，比如看gowalla训练集的2号用户160号商品
                    self.train_dict[uid] = items
                    self.alive_users.append(uid)
                    # train_unique_users.append(uid)
                    # train_users.extend([uid] * len(items))
                    # train_items.extend(items)
                    self.m_items = max(self.m_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.train_size += len(items)
        with open(self.test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    uid = int(l[0])
                    items = [int(i) for i in l[1:]]
                    items.sort()
                    self.test_dict[uid] = items
                    # test_unique_users.append(uid)
                    # test_users.extend([uid] * len(items))
                    # test_items.extend(items)
                    self.m_items = max(self.m_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.test_size += len(items)
                    
        self.m_items += 1
        self.n_users += 1
        self.test2valid_test()
        # self.alive_users = np.array(alive_users)
        # 邻接矩阵构建
        # self.interaction_matrix = sp.csr_matrix((np.ones(len(self.train_users)), (self.train_users, self.train_items)),
        #                               shape=(self.n_users, self.m_items))
        # self.R = self.R.tocsr()
        self.R = sp.dok_matrix((self.n_users, self.m_items))
        for (uid, items) in self.train_dict.items():
            for item in items:
                self.R[uid, item] = 1.
        self.norm_adj = self.get_norm_adj()
        # self.train_unique_users = np.array(train_unique_users)
        # self.train_users = np.array(train_users)
        # self.train_items = np.array(train_items)
        # self.test_unique_users = np.array(test_unique_users)
        # self.test_users = np.array(test_users)
        # self.test_items = np.array(test_items)

        # 打印数据集信息
        print(f"{self.train_size} interactions for training")
        print(f"{self.test_size} interactions for testing")
        print(f"{config.dataset_name} Sparsity : {(self.train_size + self.test_size) / self.n_users / self.m_items}")
        print(f"{config.dataset_name} is ready to go")

    def get_norm_adj(self) -> Union[torch.Tensor, list[torch.Tensor]]:
        display.color_print("loading adjacency matrix")
        start = time()

        try:
            norm_adj = sp.load_npz(self.dataset_path + '/s_pre_adj_mat.npz')
        except :
            norm_adj = self.create_norm_adj(self.R, self.n_users + self.m_items)
            sp.save_npz(self.dataset_path + '/s_pre_adj_mat.npz', norm_adj)

        if self.a_hat_split == True:
            norm_adj = self.split_a_hat(norm_adj)
            print("done split matrix")
        else:
            norm_adj = self.convert_sp_mat_to_sp_tensor(norm_adj)
            norm_adj = norm_adj.coalesce().to(self.device)
            print("don't split the matrix")

        end = time()
        display.color_print(f"costing {end - start}s, loaded norm_mat...")
        return norm_adj
    
    def create_norm_adj(self, R, shape):
        '''
        R: 交互矩阵
        '''
        R = R.tolil()
        adj_mat = sp.dok_matrix((shape, shape), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()  # 因为有的行没有数据所以会有除以0的问题存在
        d_inv[np.isinf(d_inv)] = 0.  # 把溢出的数据都变为0
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        return norm_adj
    
    def split_a_hat(self, a):
        a_fold = []
        fold_len = (self.n_users + self.m_items) // self.a_hat_folds
        for i_fold in range(self.a_hat_folds):
            start = i_fold * fold_len
            if i_fold == self.a_hat_folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            a_fold.append(self.convert_sp_mat_to_sp_tensor(a[start:end]).coalesce().to(self.device))
        return a_fold

    def convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        # return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

class SpilitDataLoader(NormalDataLoader):
    def __init__(self, config: Config):
        super(SpilitDataLoader, self).__init__(config)
        self.shards_path = config.shards_path
        self.partition_method = config.partition_method
        self.num_shards = config.num_shards
        self.max_iters = config.max_iters


        with open(self.dataset_path + '/user_pretrain.pk', 'rb') as f:
            self.pretrain_users_embedding = pickle.load(f, encoding='latin1')
        with open(self.dataset_path + '/item_pretrain.pk', 'rb') as f:
            self.pretrain_items_embedding = pickle.load(f, encoding='latin1')
        
        # if self.partition_method == "interaction_based":
        try:
            with open(self.dataset_path + '/C-' +str(self.partition_method)+'-num_'+ str(self.num_shards)+'.pk', 'rb') as f:
                self.C = pickle.load(f)
            with open(self.dataset_path + '/C_U-' +str(self.partition_method)+'-num_'+ str(self.num_shards)+'.pk', 'rb') as f:
                self.C_U = pickle.load(f)
            with open(self.dataset_path + '/C_I-' +str(self.partition_method)+'-num_'+ str(self.num_shards)+'.pk', 'rb') as f:
                self.C_I = pickle.load(f)
        except Exception:
            if self.partition_method == "interaction_based":
                # self.C, self.C_U, self.C_I = self.interaction_based_partition()
                # self.C = self.interaction_based_partition(self.pretrain_users_embedding, self.pretrain_items_embedding)
                self.C, self.C_U, self.C_I = self.interaction_based_partition(self.pretrain_users_embedding, self.pretrain_items_embedding)
            with open(self.dataset_path + '/C-' +str(self.partition_method)+'-num_'+ str(self.num_shards)+'.pk', 'wb') as f:
                pickle.dump(self.C, f)
            with open(self.dataset_path + '/C_U-' +str(self.partition_method)+'-num_'+ str(self.num_shards)+'.pk', 'wb') as f:
                pickle.dump(self.C_U, f)
            with open(self.dataset_path + '/C_I-' +str(self.partition_method)+'-num_'+ str(self.num_shards)+'.pk', 'wb')as f:
                pickle.dump(self.C_I, f)
        self.n_C = []
        for i in range(len(self.C)):
            t = 0
            for j in self.C[i]:
                t += len(self.C[i][j])
            self.n_C.append(t)

        '''
        C: 元素为字典的列表，每个字典代表一个分块中的交互，字典的key是用户，value是用户的item集合
        C_I: 元素为列表的列表，每个列表代表一个分块中包含的所有用户
        C_U: 元素为列表的列表，每个列表代表一个分块中包含的所有item
        n_C: 元素为整数的列表，每个整数代表一个分块中的交互数
        '''
        # self.norm_adjs = self.get_norm_adjs()
        self.__get_sub_dataloaders(config)

        print('training nums of each local data:')
        print(self.n_C)


    def rebuild(self, uidW, iidW, config):
        self.pretrain_users_embedding = uidW    
        self.pretrain_items_embedding = iidW
        # self.C = self.interaction_based_partition(self.pretrain_users_embedding, self.pretrain_items_embedding)
        self.C, self.C_U, self.C_I = self.interaction_based_partition(self.pretrain_users_embedding, self.pretrain_items_embedding)
        with open(self.dataset_path + '/C-' +str(self.partition_method)+'-num_'+ str(self.num_shards)+'.pk', 'wb') as f:
            pickle.dump(self.C, f)
        with open(self.dataset_path + '/C_U-' +str(self.partition_method)+'-num_'+ str(self.num_shards)+'.pk', 'wb') as f:
            pickle.dump(self.C_U, f)
        with open(self.dataset_path + '/C_I-' +str(self.partition_method)+'-num_'+ str(self.num_shards)+'.pk', 'wb')as f:
            pickle.dump(self.C_I, f)
        self.n_C = []
        for i in range(len(self.C)):
            t = 0
            for j in self.C[i]:
                t += len(self.C[i][j])
            self.n_C.append(t)
        self.__get_sub_dataloaders_cl(config)
        print('training nums of each local data:')
        print(self.n_C)

    def get_norm_adjs(self):
        display.color_print("loading adjacency matrix")
        start = time()

        norm_adjs = []
        for id_shard in range(self.num_shards):
            adj_path = self.shards_path + '/norm_adj_' + str(id_shard)+'.npz'
            try:
                norm_adj = sp.load_npz(adj_path)
            except:
                R = sp.dok_matrix((self.n_users, self.m_items), dtype=np.float32)
                for u in self.C[id_shard]:
                    for i in self.C[id_shard][u]:
                        R[u, i] = 1.
                norm_adj = self.create_norm_adj(R, self.n_users + self.m_items)
                sp.save_npz(adj_path, norm_adj)

            if self.a_hat_split == True:
                norm_adj = self.split_a_hat(norm_adj)
            else:
                norm_adj = self.convert_sp_mat_to_sp_tensor(norm_adj)
                norm_adj = norm_adj.coalesce().to(self.device)
            norm_adjs.append(norm_adj)

        end = time()
        display.color_print(f"costing {end - start}s, loaded norm_mat...")
        return norm_adjs
    
    def get_norm_adjs_cl(self, train_dicts):
        norm_adjs = []
        for id_shard in range(self.num_shards):
            R = sp.dok_matrix((self.n_users, self.m_items), dtype=np.float32)
            for u in train_dicts[id_shard]:
                for i in train_dicts[id_shard][u]:
                    R[u, i] = 1.
            norm_adj = self.create_norm_adj(R, self.n_users + self.m_items)

            if self.a_hat_split == True:
                norm_adj = self.split_a_hat(norm_adj)
            else:
                norm_adj = self.convert_sp_mat_to_sp_tensor(norm_adj)
                norm_adj = norm_adj.coalesce().to(self.device)
            norm_adjs.append(norm_adj)
        return norm_adjs

    def __get_sub_dataloaders_cl(self, config):
        norm_adjs = self.get_norm_adjs_cl(self.C)
        self.sub_dataloaders = []
        for id_shard in range(self.num_shards):
            sub_dataloader = BasicDataLoader(config)
            sub_dataloader.n_users = self.n_users
            sub_dataloader.m_items = self.m_items
            sub_dataloader.train_dict = self.C[id_shard]
            sub_dataloader.alive_users = list(sub_dataloader.train_dict.keys())
            sub_dataloader.valid_dict = self.valid_dict
            sub_dataloader.test_dict = self.test_dict
            sub_dataloader.train_size = self.n_C[id_shard]
            sub_dataloader.norm_adj = norm_adjs[id_shard]
            sub_dataloader.dom_dataloader = self  # 为sub_dataloader添加dom_dataloader, 用于sub-model测试时删除全局的训练集中出现的商品
            self.sub_dataloaders.append(sub_dataloader)


    def __get_sub_dataloaders(self, config):
        norm_adjs = self.get_norm_adjs()
        self.sub_dataloaders = []
        for id_shard in range(self.num_shards):
            sub_dataloader = BasicDataLoader(config)
            sub_dataloader.n_users = self.n_users
            sub_dataloader.m_items = self.m_items
            sub_dataloader.train_dict = self.C[id_shard]
            sub_dataloader.alive_users = list(sub_dataloader.train_dict.keys())
            sub_dataloader.valid_dict = self.valid_dict
            sub_dataloader.test_dict = self.test_dict
            sub_dataloader.train_size = self.n_C[id_shard]
            sub_dataloader.norm_adj = norm_adjs[id_shard]
            sub_dataloader.dom_dataloader = self  # 为sub_dataloader添加dom_dataloader, 用于sub-model测试时删除全局的训练集中出现的商品
            self.sub_dataloaders.append(sub_dataloader)

    def interaction_based_partition(self, uidW: np.ndarray, iidW: np.ndarray):
        # with open(self.dataset_path + '/user_pretrain.pk', 'rb') as f:
        #     uidW = pickle.load(f, encoding='latin1')
        # with open(self.dataset_path + '/item_pretrain.pk', 'rb') as f:
        #     iidW = pickle.load(f, encoding='latin1')

        # get_data_interactions_1
        data = []
        for i in self.train_dict:
            for j in self.train_dict[i]:
                data.append([i, j])
        # Randomly select k centroids
        max_data = 1.2 * len(data) // self.num_shards
        centroids = random.sample(data, self.num_shards)
        # centro emb
        centroembs = []
        for i in range(self.num_shards):
            temp_u = uidW[centroids[i][0]]
            temp_i = iidW[centroids[i][1]]
            centroembs.append([temp_u, temp_i])

        for _ in range(self.max_iters):
            C = [{} for i in range(self.num_shards)]
            C_num = [0 for i in range(self.num_shards)]
            Scores = {}
            for i in range(len(data)):
                for j in range(self.num_shards):
                    score_u = self.eucli_dist2(uidW[data[i][0]],centroembs[j][0])
                    score_i = self.eucli_dist2(iidW[data[i][1]],centroembs[j][1])
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
            for i in range(self.num_shards):
                temp_u = []
                temp_i = []
                for j in C[i].keys():
                    for l in C[i][j]:
                        temp_u.append(uidW[j])
                        temp_i.append(iidW[l])
                centroembs_next.append([np.mean(temp_u), np.mean(temp_i)])

            loss = 0.0
            for i in range(self.num_shards):
                score_u = self.eucli_dist2(centroembs_next[i][0], centroembs[i][0])
                score_i = self.eucli_dist2(centroembs_next[i][1], centroembs[i][1])
                loss += (score_u * score_i)
            centroembs = centroembs_next
            # for i in range(self.num_shards):
            #     print(C_num[i])
            print(C_num)
            print(_, loss)
        users = [[] for i in range(self.num_shards)]
        items = [[] for i in range(self.num_shards)]
        for i in range(self.num_shards):
            # dict.keys()或者dict.values()返回的是dict_keys对象，dict_items()返回的是dict_items对象
            # dict_keys对象和dict_items对象都是可迭代的，更高效，但是不能序列化，pickle会出错
            # Python低版本中不是返回dict_keys对象或者dict_items对象，不存在问题
            users[i] = list(C[i].keys())  
            for j in C[i].keys():
                for l in C[i][j]:
                    if l not in items[i]:
                        items[i].append(l)

        return C, users, items
    
    def eucli_dist(self, x, y):
        return np.sqrt(np.sum(np.square(x - y)))
    
    def eucli_dist2(self, x, y):
        return np.sum(np.square(x - y))
