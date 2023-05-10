import numpy as np
import random


def sample(n_users, m_items, alive_users, train_dict, train_batch_size, neg_ratio = 1):
    if train_batch_size <= n_users:
        users = random.sample(alive_users, train_batch_size)
    else:
        users = [random.choice(alive_users) for _ in range(train_batch_size)]


    def sample_pos_items_for_u(u, num):
        pos_items = train_dict[u]
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
            neg_id = np.random.randint(low=0, high=m_items, size=1)[0]
            if neg_id not in train_dict[u] and neg_id not in neg_items:
                neg_items.append(neg_id)
        return neg_items

    pos_items, neg_items = [], []
    for u in users:
        pos_items += sample_pos_items_for_u(u, 1)
        neg_items += sample_neg_items_for_u(u, 1)
    return users, pos_items, neg_items

# try:
#     from cppimport import imp_from_filepath
#     from os.path import join, dirname
#     path = join(dirname(__file__), "sources/sampling.cpp")
#     sampling = imp_from_filepath(path)
#     sampling.seed(config.seed)
#     sample_ext = True
# except:
#     display.color_print("Cpp extension not loaded")
#     sample_ext = False

def UniformSample_original(train_dict, train_size, n_users, m_items, neg_ratio = 1):
    # allPos = dataloader.train_list
    # start = time()
    # if sample_ext:
    #     S = sampling.sample_negative(dataset.n_users, dataset.m_items,
    #                                  dataset.trainDataSize, allPos, neg_ratio)
    # else:
    #     S = UniformSample_original_python(dataset)
    S = UniformSample_original_python(train_dict, train_size, n_users, m_items)
    return S

def UniformSample_original_python(train_dict, train_size, n_users, m_items):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """

    user_num = train_size
    users = np.random.randint(0, n_users, user_num)  # 直接抽train_size个用户?
    S = []
    for user in users:
        pos_items = train_dict.get(user)  # user在训练集中可能没有商品，在字典中没有对应的key
        # posForUser = allPos[user]
        if pos_items is None:
            continue
        indexes = np.random.randint(0, len(pos_items))
        pos_item = pos_items[indexes]
        while True:
            neg_item = np.random.randint(0, m_items)
            if neg_item in pos_items:
                continue
            else:
                break
        S.append([user, pos_item, neg_item])
    return np.array(S)
