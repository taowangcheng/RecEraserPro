import numpy as np
import torch
import multiprocessing
from functools import partial
from time import time

import utils
from utils import timer
from dataloaders import BasicDataLoader
from models import BasicModel
from loss import BPRLoss
from configuration import Config
import display
import metrics


def bpr_train_one_epoch(epoch, dataloader, model, bpr_loss, config, neg_k=1):
    return bpr_train_one_epoch1(epoch, dataloader, model, bpr_loss, config, neg_k=1)

def train_test(dataloader: BasicDataLoader, model: BasicModel, config: Config, epochs: int):
    bpr_loss = BPRLoss(model, config)
    # 训练前测试
    display.color_print("[TEST]EPOCH[0]")
    results = bpr_test(0, dataloader, model, config)
    print(results)

    # early stopping strategy:
    cur_best_pre_0 = results['recall'][0]
    stopping_step = 0


    for epoch in range(epochs):
        output_information = bpr_train_one_epoch(epoch, dataloader, model, bpr_loss, config, neg_k=1)
        print(f'EPOCH[{epoch + 1}/{config.epochs}] {output_information}')
        if (epoch + 1) % 10 == 0:
            display.color_print(f"[TEST]EPOCH[{epoch + 1}]")
            results = bpr_test(epoch + 1, dataloader, model, config)
            print(results)
            cur_best_pre_0, stopping_step, should_stop = early_stopping(results['recall'][0], cur_best_pre_0,
                                                stopping_step, expected_order='acc', flag_step=10)
            if should_stop:
                torch.save(model.state_dict(), config.weight_filename)  # 这里要保存吗?
                break
        torch.save(model.state_dict(), config.weight_filename)


def multi_train_test(dataloader: BasicDataLoader, model: BasicModel, config: Config):
    # sub-model训练
    for id_shard in range(config.num_shards):
        sub_model = model.sub_models[id_shard]
        display.color_print(f"shard {id_shard} start training and testing")
        train_test(dataloader.sub_dataloaders[id_shard], sub_model, config, config.epochs)
    # 聚合部分训练
    display.color_print(f"aggregator start training and testing")
    train_test(dataloader, model, config, config.attention_epochs)

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=10):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']
    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

def bpr_train_one_epoch1(epoch, dataloader: BasicDataLoader, model, bpr_loss: BPRLoss, config: Config, neg_k=1):
    # 一个eopch抽样训练集大小的三元组，然后进行batch训练
    model.train()
    with timer(name="Sample"):
        S = dataloader.sample1()
        # S = sampler.UniformSample_original(dataloader.train_dict, dataloader.train_size, dataloader.n_users, dataloader.m_items)
    users = torch.Tensor(S[:, 0]).long()
    pos_items = torch.Tensor(S[:, 1]).long()
    neg_items = torch.Tensor(S[:, 2]).long()
    users = users.to(config.device)
    pos_items = pos_items.to(config.device)
    neg_items = neg_items.to(config.device)
    users, pos_items, neg_items = utils.shuffle(users, pos_items, neg_items)

    total_batch = len(users) // config.train_batch_size + 1
    average_loss = 0.
    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(utils.minibatch(config.train_batch_size, users, pos_items, neg_items)):
        batch_loss = bpr_loss.step(batch_users, batch_pos, batch_neg)
        average_loss += batch_loss
        if config.tensorboard_writer:
            # python3中/不会下取整
            config.tensorboard_writer.add_scalar(f'BPRLoss/BPR', batch_loss, epoch * int(len(users) / config.train_batch_size) + batch_i + 1)
    aver_loss = average_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"

def bpr_train_one_epoch2(dataloader: BasicDataLoader, model: BasicModel, config: Config, bpr_loss: BPRLoss):
    for epoch in range(config.epochs):
        start = time()
        average_loss = 0.0
        n_batch = dataloader.train_size // config.train_batch_size + 1
        for idx in range(n_batch):
            # btime= time()
            users, pos_items, neg_items = dataloader.sample2()
            users = torch.Tensor(users).long().to(model.device)
            pos_items = torch.Tensor(pos_items).long().to(model.device)
            neg_items = torch.Tensor(neg_items).long().to(model.device)
            batch_loss = bpr_loss.step(users, pos_items, neg_items)
            average_loss += batch_loss
            if config.tensorboard_writer:
                # python3中/不会下取整
                config.tensorboard_writer.add_scalar(f'BPRLoss/BPR', batch_loss, epoch * n_batch + idx + 1)

        aver_loss = average_loss / n_batch
        epoch_time = time() - start
        return f"loss{aver_loss:.3f}-epoch_time{epoch_time}"
    
def test_one_batch(x, topks):
    """
    X: (sorted_items, groundTrue)
    sorted_items: [test_batch_size, MaxTopK]，已经按照打分排序的item，里面的item是id，numpy格式
    groundTrue: [test_batch_size, item_num]，groundTrue中的item是id，变长的嵌套list
    计算测试阶段一个test_batch的recall, precision, ndcg
    返回的是一个test_batch的TopK的recall, precision, ndcg，没有对batch进行平均
    """
    sorted_items = x[0].numpy()
    groundTrue = x[1]
    r = metrics.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = metrics.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(metrics.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def bpr_test(epoch, dataloader: BasicDataLoader, model: BasicModel, config: Config):
    # eval mode with no dropout
    model = model.eval()
    train_dict = dataloader.train_dict
    test_dict = dataloader.test_dict

    max_K = max(config.topks)

    results = {'precision': np.zeros(len(config.topks)),
               'recall': np.zeros(len(config.topks)),
               'ndcg': np.zeros(len(config.topks))}
    with torch.no_grad():
        users = list(test_dict.keys())
        try:
            assert config.test_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundtruth_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // config.test_batch_size + 1
        for batch_users in utils.minibatch(config.test_batch_size, users):
            # 测试集中的user在训练集中可能没有商品，在字典中没有对应的key
            # 一般应该不会出现这种情况，训练集中应该没有孤立点
            # 用get会返回None，再去除None
            pos_items = [train_dict.get(u) for u in batch_users] 
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(pos_items):
                if items is None:
                    continue
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            groundtruth_items = [test_dict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(config.device)
            rating = model.get_users_rating(batch_users_gpu) # 因为模型在GPU上，不能跨设备运算
            # rating[exclude_index, exclude_items] = -(1<<10)
            rating[exclude_index, exclude_items] = -1 # 因为评分经过了sigmoid，所以这里设置为-1
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundtruth_list.append(groundtruth_items)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundtruth_list)
        if config.multicore:
            cores = multiprocessing.cpu_count() // 2
            pool = multiprocessing.Pool(cores)
            # pre_results = pool.map(test_one_batch, (X, topks)) # 只能传入一个参数
            partial_work = partial(test_one_batch, topks=config.topks) # 提取x作为partial函数的输入变量
            pre_results = pool.map(partial_work, X)
            pool.close()
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x, config.topks))
        # scale = float(test_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if config.tensorboard_writer:
            config.tensorboard_writer.add_scalars(f'Test/Recall@{config.topks}',
                          {str(config.topks[i]): results['recall'][i] for i in range(len(config.topks))}, epoch)
            config.tensorboard_writer.add_scalars(f'Test/Precision@{config.topks}',
                          {str(config.topks[i]): results['precision'][i] for i in range(len(config.topks))}, epoch)
            config.tensorboard_writer.add_scalars(f'Test/NDCG@{config.topks}',
                          {str(config.topks[i]): results['ndcg'][i] for i in range(len(config.topks))}, epoch)
            
        # print(results)
        return results
    