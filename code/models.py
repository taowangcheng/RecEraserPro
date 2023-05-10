import torch
from torch import nn
from torch import optim
import numpy as np
import multiprocessing
from functools import partial
from time import time
from typing import Union

import utils
import metrics
import display
from dataloaders import BasicDataLoader, SpilitDataLoader
from configuration import Config
from evaluator import eval_score_matrix_foldout

class BasicModel(nn.Module):
    def __init__(self, config: Config, dataloader: BasicDataLoader):
        super(BasicModel, self).__init__()
        # 模型和数据集配置
        self.id: Union[None, int]
        self.n_users = dataloader.n_users
        self.m_items = dataloader.m_items
        self.norm_adj = dataloader.norm_adj
        # 超参数配置
        self.epochs: int
        self.topks = config.topks
        self.embedding_size = config.embedding_size
        self.layers = config.layers
        self.lr = config.lr
        self.decay = config.decay
        self.dropout = config.dropout
        self.keep_prob = config.keep_prob
        # 设备配置
        self.device = config.device
        # 划分配置
        self.num_shards: int
        self.sub_models: None or list[BasicModel]
        # 其它配置
        self.a_hat_split = config.a_hat_split
        self.multicore = config.multicore
        self.pretrain = config.pretrain
        self.weight_filename = config.weight_filename  # sub-model要区分
        self.tensorboard_writer = config.tensorboard_writer
        self.verbose = config.verbose

    def __init_weight(self):
        raise NotImplementedError
    
    # 函数名换成__init_loss会报错, 不知道为什么
    # raise AttributeError("'{}' object has no attribute '{}'".format(
    # AttributeError: 'LightGCN' object has no attribute '_LightGCN__init_loss'
    def loss(self):
        self.opt = optim.Adam(self.parameters(), lr=self.lr) 

    def step(self, users, pos_items, neg_items):
        loss, reg_loss = self.bpr_loss(users, pos_items, neg_items)
        reg_loss = reg_loss * self.decay
        loss = loss + reg_loss
        # 优化模型参数
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.cpu().item()  # 返回loss的值
    
    def train_test(self, dataloader: BasicDataLoader):
        self.loss()
        raise NotImplementedError

    def bpr_loss(self, users, pos_items, neg_items):
        raise NotImplementedError
    
    def get_users_rating(self, users):
        raise NotImplementedError
    
    def __dropout_adj(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        # g = torch.sparse.FloatTensor(index.t(), values, size)
        g = torch.sparse_coo_tensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.a_hat_split:
            norm_adj = []
            for adj in self.norm_adj:
                norm_adj.append(self.__dropout_adj(adj, keep_prob))
        else:
            norm_adj = self.__dropout_adj(self.norm_adj, keep_prob)
        return norm_adj
        
    def computer(self, init_users_embeddings, init_items_embeddings):
        """
        propagate methods for lightGCN
        """       
        all_embeddings = torch.cat([init_users_embeddings, init_items_embeddings])
        final_embeddings = [all_embeddings]
        if self.dropout:
            if self.training:
                # print("droping")
                norm_adj = self.__dropout(self.keep_prob)
            else:
                norm_adj = self.norm_adj
        else:
            norm_adj = self.norm_adj
        
        for _ in range(self.layers):
            if self.a_hat_split:
                temp_embeddings = []
                for f in range(len(norm_adj)):
                    temp_embeddings.append(torch.sparse.mm(norm_adj[f], all_embeddings))
                all_embeddings = torch.cat(temp_embeddings, dim=0)
            else:
                # display.color_print("norm_adj.shape" + str(norm_adj.shape))
                # display.color_print("all_embeddings.shape" + str(all_embeddings.shape))
                all_embeddings = torch.sparse.mm(norm_adj, all_embeddings)
            final_embeddings.append(all_embeddings)
        final_embeddings = torch.stack(final_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        return torch.split(final_embeddings, [self.n_users, self.m_items])

    def train_test1(self, dataloader: BasicDataLoader):
        sample = dataloader.sample1
        bpr_test = self.bpr_test1
        # 训练前测试
        # display.color_print("[TEST]EPOCH[0]")
        # results = self.bpr_test(0, dataloader)
        # results = bpr_test(0, dataloader)
        # print(results)

        # early stopping strategy:
        # cur_best_pre_0 = results['recall'][0]
        best_value = 0.0
        stopping_step = 0
        for epoch in range(self.epochs):
            self.train()
            start = time()
            sum_loss = 0.0
            # S = dataloader.sample1()
            S = sample()
            users = torch.Tensor(S[:, 0]).long()
            pos_items = torch.Tensor(S[:, 1]).long()
            neg_items = torch.Tensor(S[:, 2]).long()
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)
            users, pos_items, neg_items = utils.shuffle(users, pos_items, neg_items)
            n_batch = len(users) // dataloader.train_batch_size + 1
            for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(utils.minibatch(dataloader.train_batch_size, users, pos_items, neg_items)):
                batch_loss = self.step(batch_users, batch_pos, batch_neg)
                sum_loss += batch_loss
                if self.tensorboard_writer:
                    # python3中/不会下取整
                    self.tensorboard_writer.add_scalar(f'BPRLoss/BPR', batch_loss, epoch * n_batch + batch_i + 1)
            average_loss = sum_loss / n_batch
            time_loss = time() - start
            train_info = f"EPOCH[{epoch + 1}/{self.epochs}] loss{average_loss:.3f}-time{time_loss:.3f}s"
            if self.id is not None:
                train_info = f"id_model[{self.id}/{self.num_shards}] " + train_info
            print(train_info)

            if (epoch + 1) % self.verbose == 0:
                # 验证集
                results = bpr_test(epoch + 1, dataloader, test_flag=False)
                valid_info = f"[Valid] " + str(results)  # f"[Valid]EPOCH[{epoch + 1}]"
                print(valid_info)
                if (best_value < results['recall'][1]):
                    torch.save(self.state_dict(), self.weight_filename)
                    display.color_print(f"save model to {self.weight_filename}")
                best_value, stopping_step, should_stop = self.early_stopping(results['recall'][1], best_value,
                                    stopping_step, expected_order='acc', flag_step=100)
                if should_stop:
                    break
                # 测试集
                results = bpr_test(epoch + 1, dataloader, test_flag=True)
                test_info = f"[Test] " + str(results)
                print(test_info)

            # if (epoch + 1) % self.verbose == 0:
            #     display.color_print(f"[TEST]EPOCH[{epoch + 1}]")
            #     # results = self.bpr_test1(epoch + 1, dataloader)
            #     results = bpr_test(epoch + 1, dataloader)
            #     print(results)
            #     cur_best_pre_0, stopping_step, should_stop = self.early_stopping(results['recall'][0], cur_best_pre_0,
            #                                         stopping_step, expected_order='acc', flag_step=10)
            #     if should_stop:
            #         torch.save(self.state_dict(), self.weight_filename)  # 这里要保存吗?
            #         break
            # torch.save(self.state_dict(), self.weight_filename)

    def train_test2(self, dataloader: BasicDataLoader):
        sample = dataloader.sample2
        bpr_test = self.bpr_test2
        # 训练前测试
        # display.color_print("[TEST]EPOCH[0]")
        # # results = self.bpr_test2(0, dataloader)
        # results = bpr_test(0, dataloader)
        # print(results)

        # early stopping strategy:
        best_value = 0.0
        stopping_step = 0
        n_batch = dataloader.train_size // dataloader.train_batch_size + 1
        for epoch in range(self.epochs):
            self.train()
            start = time()
            sum_loss = 0.0
            for idx in range(n_batch):
                # users, pos_items, neg_items = dataloader.sample2()
                users, pos_items, neg_items = sample()
                users = torch.Tensor(users).long().to(self.device)
                pos_items = torch.Tensor(pos_items).long().to(self.device)
                neg_items = torch.Tensor(neg_items).long().to(self.device)
                batch_loss = self.step(users, pos_items, neg_items)
                sum_loss += batch_loss
                if self.tensorboard_writer:
                    # python3中/不会下取整
                    self.tensorboard_writer.add_scalar(f'BPRLoss/BPR', batch_loss, epoch * n_batch + idx + 1)
            average_loss = sum_loss / n_batch
            time_loss = time() - start
            train_info = f"EPOCH[{epoch + 1}/{self.epochs}] loss{average_loss:.3f}-time{time_loss:.3f}s"
            if self.id is not None:
                train_info = f"id_model[{self.id}/{self.num_shards}] " + train_info
            print(train_info)
            if (epoch + 1) % self.verbose == 0:
                # 验证集
                results = bpr_test(epoch + 1, dataloader, test_flag=False)
                valid_info = f"[Valid] " + str(results)  # f"[Valid]EPOCH[{epoch + 1}]"
                print(valid_info)
                if (best_value < results['recall'][1]):
                    torch.save(self.state_dict(), self.weight_filename)
                    display.color_print(f"save model to {self.weight_filename}")
                best_value, stopping_step, should_stop = self.early_stopping(results['recall'][1], best_value,
                                    stopping_step, expected_order='acc', flag_step=100)
                if should_stop:
                    break
                # 测试集
                results = bpr_test(epoch + 1, dataloader, test_flag=True)
                test_info = f"[Test] " + str(results)
                print(test_info)
            # if (epoch + 1) % self.verbose == 0:
            #     display.color_print(f"[TEST]EPOCH[{epoch + 1}]")
            #     # users_to_test = list(data_generator.test_set.keys())
            #     # ret = test(sess, model, users_to_test, drop_flag=False)
            #     # results = self.bpr_test2(epoch + 1, dataloader)
            #     results = bpr_test(epoch + 1, dataloader)
            #     print(results)
            #     cur_best_pre_0, stopping_step, should_stop = self.early_stopping(results['recall'][0], cur_best_pre_0,
            #                                         stopping_step, expected_order='acc', flag_step=10)
            #     if should_stop:
            #         torch.save(self.state_dict(), self.weight_filename)  # 这里要保存吗?
            #         break
            # torch.save(self.state_dict(), self.weight_filename)

    def test_one_batch(self, x):
        """
        X: (sorted_items, groundTrue)
        sorted_items: [test_batch_size, MaxTopK], 已经按照打分排序的item, 里面的item是id, numpy格式
        groundTrue: [test_batch_size, item_num], groundTrue中的item是id, 变长的嵌套list
        计算测试阶段一个test_batch的recall, precision, ndcg
        返回的是一个test_batch的TopK的recall, precision, ndcg, 没有对batch进行平均
        """
        sorted_items = x[0].numpy()
        groundTrue = x[1]
        r = metrics.getLabel(groundTrue, sorted_items)
        pre, recall, ndcg = [], [], []
        for k in self.topks:  # 唯一耦合的地方
            ret = metrics.RecallPrecision_ATk(groundTrue, r, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])
            ndcg.append(metrics.NDCGatK_r(groundTrue,r,k))
        return {'recall':np.array(recall), 
                'precision':np.array(pre), 
                'ndcg':np.array(ndcg)}
                
    def bpr_test1(self, epoch, dataloader: BasicDataLoader, test_flag=True): # , 
        # eval mode with no dropout
        self.eval()
        if dataloader.dom_dataloader is not None:
            train_dict = dataloader.dom_dataloader.train_dict
        else:
            train_dict = dataloader.train_dict
        if test_flag:
            test_dict = dataloader.test_dict
        else:
            test_dict = dataloader.valid_dict
        test_batch_size = dataloader.test_batch_size
        max_K = max(self.topks)
        results = {'precision': np.zeros(len(self.topks)),
                'recall': np.zeros(len(self.topks)),
                'ndcg': np.zeros(len(self.topks))}
        with torch.no_grad():
            users = list(test_dict.keys())
            try:
                assert test_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundtruth_list = []
            total_batch = len(users) // test_batch_size + 1
            for batch_users in utils.minibatch(test_batch_size, users):
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
                batch_users_gpu = batch_users_gpu.to(self.device)
                rating = self.get_users_rating(batch_users_gpu) # 因为模型在GPU上，不能跨设备运算
                # rating[exclude_index, exclude_items] = -(1<<10)
                rating[exclude_index, exclude_items] = -1 # 因为评分经过了sigmoid，所以这里设置为-1
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundtruth_list.append(groundtruth_items)
            assert total_batch == len(users_list)
            X = zip(rating_list, groundtruth_list)
            if self.multicore:
                cores = multiprocessing.cpu_count() // 2
                pool = multiprocessing.Pool(cores)
                # pre_results = pool.map(test_one_batch, (X, topks)) # 只能传入一个参数
                # partial_work = partial(self.test_one_batch, topks=self.topks) # 提取x作为partial函数的输入变量
                # pre_results = pool.map(partial_work, X)
                # pre_results = pool.starmap(self.test_one_batch, zip(X, [self.topks] * total_batch)) # 传入多个参数, 这样应该可以
                pre_results = pool.map(self.test_one_batch, X)
                pool.close()
            else:
                pre_results = []
                for x in X:
                    pre_results.append(self.test_one_batch(x))
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalars(f'Test/Recall@{self.topks}',
                            {str(self.topks[i]): results['recall'][i] for i in range(len(self.topks))}, epoch)
                self.tensorboard_writer.add_scalars(f'Test/Precision@{self.topks}',
                            {str(self.topks[i]): results['precision'][i] for i in range(len(self.topks))}, epoch)
                self.tensorboard_writer.add_scalars(f'Test/NDCG@{self.topks}',
                            {str(self.topks[i]): results['ndcg'][i] for i in range(len(self.topks))}, epoch)
            return results

    def bpr_test2(self, epoch, dataloader: BasicDataLoader, test_flag=True):
        self.eval()
        if dataloader.dom_dataloader is not None:
            train_dict = dataloader.dom_dataloader.train_dict
        else:
            train_dict = dataloader.train_dict
        if test_flag:
            test_dict = dataloader.test_dict
        else:
            test_dict = dataloader.valid_dict
        test_batch_size = dataloader.test_batch_size
        # B: batch size
        # N: the number of items
        top_show = np.sort(self.topks)
        max_top = max(top_show)
        result = {'precision': np.zeros(len(self.topks)), 'recall': np.zeros(len(self.topks)), 'ndcg': np.zeros(len(self.topks))}
        # u_batch_size = dataloader.test_batch_size
        users = list(test_dict.keys())
        n_batch = len(users) // test_batch_size + 1
        count = 0
        all_result = []
        with torch.no_grad():
            for batch_id in range(n_batch):
                start = batch_id * test_batch_size
                end = (batch_id + 1) * test_batch_size
                batch_users = users[start: end]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(self.device)
                batch_rating = self.get_users_rating(batch_users_gpu)
                batch_rating = batch_rating.cpu().numpy() # (B, N)
                test_items = []
                for user in batch_users:
                    test_items.append(test_dict[user])# (B, #test_items)
                # set the ranking scores of training items to -inf,
                # then the training items will be sorted at the end of the ranking list.    
                for idx, user in enumerate(batch_users):
                        # train_items_off = dataloader.train_dict[user]
                        # train_items_off = dataloader.train_dict.get(user) # RecEraser这里使用的是全局完整的train_dict
                        # if train_items_off is not None:  
                        #     batch_rating[idx][train_items_off] = -np.inf
                        train_items_off = train_dict[user]
                        batch_rating[idx][train_items_off] = -np.inf
                # rate_batch的形状是什么？答案是(B,N)，B是batch_size，N是n_items
                # test_items的形状是什么？答案是(B,#test_items)，它是一个嵌套列表，每个元素是一个列表，列表中的元素是用户的测试集中的item
                batch_result = eval_score_matrix_foldout(batch_rating, test_items, max_top)#(B,k*metric_num), max_top= 20
                count += len(batch_result)
                all_result.append(batch_result)
            assert count == len(users)
            all_result = np.concatenate(all_result, axis=0)
            final_result = np.mean(all_result, axis=0)  # mean
            final_result = np.reshape(final_result, newshape=[5, max_top])
            # final_result是一个matrix，如果要求的最大是Top50，final_result中又Top1到Top50的结果
            # 下面把top_show中的topks减一，下标从0开始，然后取出要求的topks的结果
            final_result = final_result[:, top_show-1] # 这里为什么要减1？答案是因为top_show是从1开始的，而final_result是从0开始的
            final_result = np.reshape(final_result, newshape=[5, len(top_show)])
            # final_result有5行，代表5个指标，每个指标有len(top_show)列，代表top_show中的每个topk
            result['precision'] += final_result[0]
            result['recall'] += final_result[1]
            result['ndcg'] += final_result[3]
            return result

    def early_stopping(self, log_value, best_value, stopping_step, expected_order='acc', flag_step=10):
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


class LightGCN(BasicModel):
    def __init__(self, config: Config, dataloader: BasicDataLoader):
        super(LightGCN, self).__init__(config, dataloader)
        self.epochs = config.epochs
        self.id = None
        self.__init_weight()
        
    def __init_weight(self):
        # 初始嵌入
        self.init_users_embeddings = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.embedding_size)
        self.init_items_embeddings = torch.nn.Embedding(
            num_embeddings=self.m_items, embedding_dim=self.embedding_size)
        # nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        # nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        # print('use xavier initilizer')
        # random normal init seems to be a better choice
        # when lightGCN actually don't use any non-linear activation function
        if not self.pretrain:
            nn.init.normal_(self.init_users_embeddings.weight, std=0.1)  # pytorch会使用std参数的平方
            nn.init.normal_(self.init_items_embeddings.weight, std=0.1)
            # nn.init.normal_(self.init_users_embeddings.weight, std=0.01)  # pytorch会使用std参数的平方
            # nn.init.normal_(self.init_items_embeddings.weight, std=0.01)
            # display.color_print('use NORMAL distribution initilizer')
        else:
            display.color_print('use pretarined data')
        self.sigmoid = nn.Sigmoid()
        print(f"LightGCN is already to go(dropout:{self.dropout})")
        
    
    def train_test(self, dataloader: BasicDataLoader):
        self.loss()
        self.train_test2(dataloader)

    def get_users_rating(self, users):
        users = users.long()
        init_users_embeddings = self.init_users_embeddings.weight
        init_items_embeddings = self.init_items_embeddings.weight
        final_users_embeddings, final_items_embeddings = self.computer(
            init_users_embeddings, init_items_embeddings)
        
        users_embeddings = final_users_embeddings[users]
        items_embeddings = final_items_embeddings
        rating = self.sigmoid(torch.matmul(users_embeddings, items_embeddings.t()))
        return rating
    
    def bpr_loss(self, users, pos_items, neg_items):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        users = users.long()
        pos_items = pos_items.long()
        neg_items = neg_items.long()

        reg_loss = 0.0
        reg_loss += self.init_users_embeddings(users).norm(2).pow(2)
        reg_loss += self.init_items_embeddings(pos_items).norm(2).pow(2)
        reg_loss += self.init_items_embeddings(neg_items).norm(2).pow(2)
        reg_loss = (1 / 2) * reg_loss / float(len(users))
        
        init_users_embeddings = self.init_users_embeddings.weight
        init_items_embeddings = self.init_items_embeddings.weight
        final_users_embeddings, final_items_embeddings = self.computer(
            init_users_embeddings, init_items_embeddings)
        users_embeddings = final_users_embeddings[users]
        pos_items_embeddings = final_items_embeddings[pos_items]
        neg_items_embeddings = final_items_embeddings[neg_items]
        pos_scores = torch.mul(users_embeddings, pos_items_embeddings)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_embeddings, neg_items_embeddings)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        users_embeddings = self.init_users_embeddings.weight
        items_embeddings = self.init_items_embeddings.weight
        final_users_embeddings, final_items_embeddings = self.computer(users_embeddings, items_embeddings)
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = final_users_embeddings[users]
        items_emb = final_items_embeddings[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

class RecEraser(BasicModel):
    def __init__(self, config: Config, dataloader: BasicDataLoader):
        super(RecEraser, self).__init__(config, dataloader)
        self.num_shards = config.num_shards
        self.attention_size = config.attention_size
        self.epochs = config.attention_epochs
        self.id = None
        self.sub_models = []
        for id_shard in range(self.num_shards):
            sub_model = LightGCN(config, dataloader.sub_dataloaders[id_shard])
            sub_model.id = id_shard
            sub_model.num_shards = self.num_shards
            f"id_model[{self.id}/{self.num_shards}] "
            sub_model.weight_filename = sub_model.weight_filename[:-4] + f"id_[{self.id}-{self.num_shards}]" + ".pth"
            self.sub_models.append(sub_model)
        self.__init_weight()
        
    def __init_weight(self):
        self.linears = nn.ModuleList([torch.nn.Linear(self.embedding_size, self.embedding_size) 
                        for _ in range(self.num_shards)])
        self.W_b_user = torch.nn.Linear(self.embedding_size, self.attention_size)
        self.H_user = torch.nn.Linear(self.attention_size, 1, bias=False)
        self.W_b_item = torch.nn.Linear(self.embedding_size, self.attention_size)
        self.H_item = torch.nn.Linear(self.attention_size, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.keep_prob = 0.7
        # RecEraser在测试的时候应该没有禁用dropout
        self.dropout_layer = torch.nn.Dropout(1 - self.keep_prob)  # torch.nn.Dropout和tf.nn.dropout的参数不同, 一个是保留率, 一个是丢弃率

        if not self.pretrain:
            for linear in self.linears:
                nn.init.normal_(linear.weight, std=0.01)  # pytorch会使用std参数的平方
                nn.init.normal_(linear.bias, std=0.01)
            # 注意大写的Tensor和小写的tensor是不一样的
            nn.init.trunc_normal_(self.W_b_user.weight, mean=0.0, std=torch.sqrt(torch.tensor(2.0 / (self.embedding_size + self.attention_size))).item())
            nn.init.constant_(self.W_b_user.bias, 0.00)
            nn.init.constant_(self.H_user.weight, 0.01)
            nn.init.trunc_normal_(self.W_b_item.weight, mean=0.0, std=torch.sqrt(torch.tensor(2.0 / (self.embedding_size + self.attention_size))).item())
            nn.init.constant_(self.W_b_item.bias, 0.00)
            nn.init.constant_(self.H_item.weight, 0.01)
            display.color_print('use initilizer')
        else:
            display.color_print('use pretarined data')
        print(f"RecEraserLightGCN is already to go(dropout:{self.dropout})")

    def train_test(self, dataloader: BasicDataLoader):
        # sub-model训练
        for id_shard in range(self.num_shards):
            sub_model = self.sub_models[id_shard]
            display.color_print(f"shard {id_shard} start training and testing")
            sub_model.train_test(dataloader.sub_dataloaders[id_shard])
        # 聚合部分训练
        display.color_print(f"aggregator start training and testing")
        for id_shard in range(self.num_shards):
            sub_model = self.sub_models[id_shard]
            sub_model.load_state_dict(torch.load(sub_model.weight_filename))
            sub_model.eval()
        self.loss()
        self.train_test2(dataloader)

    def attention(self):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        # users_embeddings = torch.empty(size=(self.num_shards, self.n_users, self.embedding_size)).to(self.device)
        # items_embeddings = torch.empty(size=(self.num_shards, self.m_items, self.embedding_size)).to(self.device)
        users_embeddings = []
        items_embeddings = []
        for id_shard in range(self.num_shards):
            sub_model = self.sub_models[id_shard]
            # users_embeddings[id_shard] = self.linears[id_shard](sub_model.init_users_embeddings.weight.detach())
            # items_embeddings[id_shard] = self.linears[id_shard](sub_model.init_items_embeddings.weight.detach())
            users_embeddings.append(self.linears[id_shard](sub_model.init_users_embeddings.weight.detach()))
            items_embeddings.append(self.linears[id_shard](sub_model.init_items_embeddings.weight.detach()))
        users_embeddings = torch.stack(users_embeddings, dim=0)
        items_embeddings = torch.stack(items_embeddings, dim=0)

        alpha_users = self.H_user(nn.functional.relu(self.W_b_user(users_embeddings)))
        alpha_users = torch.softmax(alpha_users, dim=0)
        beta_items = self.H_item(nn.functional.relu(self.W_b_item(items_embeddings)))
        beta_items = torch.softmax(beta_items, dim=0)
        agg_users_embeddings = torch.sum(torch.multiply(alpha_users, users_embeddings), dim=0)
        agg_items_embeddings = torch.sum(torch.multiply(beta_items, items_embeddings), dim=0)
        return agg_users_embeddings, agg_items_embeddings
    
    def get_users_rating(self, users):
        users = users.long()
        init_users_embeddings, init_items_embeddings = self.attention()
        final_users_embeddings, final_items_embeddings = self.computer(
            init_users_embeddings, init_items_embeddings)        
        users_embeddings = final_users_embeddings[users]
        items_embeddings = final_items_embeddings
        rating = self.sigmoid(torch.matmul(users_embeddings, items_embeddings.t()))
        return rating

    def bpr_loss(self, users, pos_items, neg_items):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        users = users.long()
        pos_items = pos_items.long()
        neg_items = neg_items.long()

        reg_loss = 0.0
        # for linear in self.linears:
        #     reg_loss += (1/2)*(linear.weight.norm().pow(2)) / float(len(users))
        # reg_loss += (1/2)*(self.W_b_user.weight.norm().pow(2)) / float(len(users))
        # reg_loss += (1/2)*(self.W_b_user.bias.norm().pow(2)) / float(len(users))
        # reg_loss += (1/2)*(self.W_b_item.weight.norm().pow(2)) / float(len(users))
        # reg_loss += (1/2)*(self.W_b_item.bias.norm().pow(2)) / float(len(users))
        # reg_loss += (1/2)*(self.H_user.weight.norm().pow(2)) / float(len(users))
        # reg_loss += (1/2)*(self.H_item.weight.norm().pow(2)) / float(len(users))

        init_users_embeddings, init_items_embeddings = self.attention()
        final_users_embeddings, final_items_embeddings = self.computer(
            init_users_embeddings, init_items_embeddings)
        users_embeddings = final_users_embeddings[users]
        pos_items_embeddings = final_items_embeddings[pos_items]
        neg_items_embeddings = final_items_embeddings[neg_items]

        users_embeddings = self.dropout_layer(users_embeddings)  # RecEraser中的trick

        pos_scores = torch.mul(users_embeddings, pos_items_embeddings)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_embeddings, neg_items_embeddings)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss, reg_loss
    
class CL(nn.Module):
    def __init__(self, config: Config, dataloader: SpilitDataLoader):
        super(CL, self).__init__()
        # 模型和数据集配置
        self.id: Union[None, int]
        self.n_users = dataloader.n_users
        self.m_items = dataloader.m_items
        self.norm_adj = dataloader.norm_adj
        # 超参数配置
        self.epochs: int
        self.topks = config.topks
        self.embedding_size = config.embedding_size
        self.layers = config.layers
        self.lr = config.lr
        self.decay = config.decay
        self.dropout = config.dropout
        self.keep_prob = config.keep_prob
        # 设备配置
        self.device = config.device
        # 划分配置
        self.num_shards: int
        self.sub_models: None or list[BasicModel]
        # 其它配置
        self.a_hat_split = config.a_hat_split
        self.multicore = config.multicore
        self.pretrain = config.pretrain
        self.weight_filename = config.weight_filename
        self.tensorboard_writer = config.tensorboard_writer
        self.verbose = config.verbose
        
        self.temp = config.temp

        self.num_shards = config.num_shards
        self.basic_model = BasicModel(config, dataloader)
        self.dataloader = dataloader
        self.pretrain_users_embedding = dataloader.pretrain_users_embedding
        self.pretrain_items_embedding = dataloader.pretrain_items_embedding
        self.pretrain_items_embedding = self.pretrain_items_embedding[:-1]

        self.__init_weight()
    

    def __init_weight(self):
        # 初始嵌入
        self.init_users_embeddings = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.embedding_size)
        self.init_items_embeddings = torch.nn.Embedding(
            num_embeddings=self.m_items, embedding_dim=self.embedding_size)
        display.color_print("n_users:" + str(self.n_users))
        display.color_print("m_items:" + str(self.m_items))
        self.init_users_embeddings.weight.data = torch.Tensor(self.pretrain_users_embedding).to(self.device)
        self.init_items_embeddings.weight.data = torch.Tensor(self.pretrain_items_embedding).to(self.device)
        display.color_print("init_users_embeddings:" + str(self.init_users_embeddings.weight.data.shape))
        display.color_print("init_items_embeddings:" + str(self.init_items_embeddings.weight.data.shape))

    def propagation(self, in_users_embeddings, in_items_embeddings, norm_adjs):
        out_users_embeddings_all = []
        out_items_embeddings_all = []
        for id_shard in range(self.num_shards):
            self.basic_model.norm_adj = norm_adjs[id_shard]
            out_users_embeddings, out_items_embeddings = self.basic_model.computer(in_users_embeddings, in_items_embeddings)
            out_users_embeddings_all.append(out_users_embeddings)
            out_items_embeddings_all.append(out_items_embeddings)
        return out_users_embeddings_all, out_items_embeddings_all
    
    def cl_loss(self, users, items, norm_adjs):
        init_users_embeddings = self.init_users_embeddings.weight
        init_items_embeddings = self.init_items_embeddings.weight
        out_users_embeddings_all, out_items_embeddings_all = self.propagation(init_users_embeddings, init_items_embeddings, norm_adjs)

        # 随机选两个出来
        id_shard1, id_shard2 = np.random.randint(0, self.num_shards - 1, size=2)
        out_users_embeddings1 = out_users_embeddings_all[id_shard1]
        out_items_embeddings1 = out_items_embeddings_all[id_shard1]
        out_users_embeddings2 = out_users_embeddings_all[id_shard2]
        out_items_embeddings2 = out_items_embeddings_all[id_shard2]

        cl_loss = 0.0

        u_mask = (torch.rand(len(users)) > 0.5).float().cuda(self.device)
        users_view1 = nn.functional.normalize(out_users_embeddings1[users], p=2, dim=1)
        users_view2 = nn.functional.normalize(out_users_embeddings2[users], p=2, dim=1)
        pos_score = torch.exp((users_view1 * users_view2).sum(1) / self.temp)
        neg_score = torch.exp(users_view1 @ users_view2.T / self.temp).sum(1)
        users_loss = ((-1 * torch.log(pos_score/(neg_score+1e-8) + 1e-8)) * u_mask).sum()
        cl_loss += users_loss

        i_mask = (torch.rand(len(items)) > 0.5).float().cuda(self.device)
        items_view1 = nn.functional.normalize(out_items_embeddings1[items], p=2, dim=1)
        items_view2 = nn.functional.normalize(out_items_embeddings2[items], p=2, dim=1)
        pos_score = torch.exp((items_view1 * items_view2).sum(1) / self.temp)
        neg_score = torch.exp(items_view1 @ items_view2.T / self.temp).sum(1)
        items_loss = ((-1 * torch.log(pos_score/(neg_score+1e-8) + 1e-8))*i_mask).sum()
        cl_loss += items_loss
        
        return cl_loss

    def step(self, users, items, norm_adjs):
        loss = self.cl_loss(users, items, norm_adjs)
        # 优化模型参数
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.cpu().item()  # 返回loss的值

    def loss(self):
        self.opt = optim.Adam(self.parameters(), lr=self.lr) 

    def train_test(self, dataloader: BasicDataLoader):
        self.epochs = 10
        self.loss()
        sample = dataloader.sample2
        n_batch = dataloader.train_size // dataloader.train_batch_size + 1
        for epoch in range(self.epochs):
            self.train()
            start = time()

            in_users_embeddings = self.init_users_embeddings.weight.detach().cpu().numpy()
            in_items_embeddings = self.init_items_embeddings.weight.detach().cpu().numpy()
            sub_train_dicts: list[dict] = self.dataloader.interaction_based_partition(in_users_embeddings, in_items_embeddings)
            norm_adjs = self.dataloader.get_norm_adjs_cl(sub_train_dicts)

            sum_loss = 0.0
            for idx in range(n_batch):
                users, pos_items, neg_items = sample()
                items = set()
                items = items.union(pos_items)
                items = items.union(neg_items)
                items = list(items)
                users = torch.Tensor(users).long().to(self.device)
                items = torch.Tensor(items).long().to(self.device)
                batch_loss = self.step(users, items, norm_adjs)
                sum_loss += batch_loss
                if self.tensorboard_writer:
                    # python3中/不会下取整
                    self.tensorboard_writer.add_scalar(f'BPRLoss/BPR', batch_loss, epoch * n_batch + idx + 1)
            average_loss = sum_loss / n_batch
            time_loss = time() - start
            train_info = f"CL_MODEL: EPOCH[{epoch + 1}/{self.epochs}] loss{average_loss:.3f}-time{time_loss:.3f}s"
            print(train_info)
        