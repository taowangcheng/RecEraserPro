import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")

    # 模型配置&&数据集配置
    parser.add_argument('--model', type=str, default='LightGCN', help='recommendation-model, support [LightGCN]')
    parser.add_argument('--dataset', type=str, default='gowalla', help="available datasets: [ml-1m, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--dataloader', type=str, default='normal', help="available dataloaders: [normal, spilit]")

    # 超参数配置
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--topks', nargs='?', default="[20]", help="@k test list")
    parser.add_argument('--embedding_size', type=int, default=64, help="the embedding size of lightGCN")
    parser.add_argument('--train_batch_size', type=int, default=2048, help="the batch size for bpr loss training procedure")
    parser.add_argument('--test_batch_size', type=int, default=100, help="the batch size of users for testing")
    parser.add_argument('--layers', type=int, default=3, help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float, default=0.001, help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-4, help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=bool, default=False, help="using the dropout or not")
    parser.add_argument('--keep_prob', type=float, default=0.6, help="1 - dropuot ratio")
    
    # 划分配置
    parser.add_argument('--data_partition', type=bool, default=False, help="whether we split the data into different shards")
    parser.add_argument('--partition_method', type=str, default='interaction_based', help='the method of data partition, support [random, interaction_based, ...]')
    parser.add_argument('--num_shards', type=int, default=10, help="the number of shards")
    parser.add_argument('--max_iters', type=int, default=50, help="the number of iterations for data partition")
    parser.add_argument('--attention_size', type=int, default=32, help="the size of attention")
    parser.add_argument('--attention_epochs', type=int, default=50, help="the number of epochs for attention training")

    parser.add_argument('--temp', type=float, default=0.5, help="the temperature of cl")
    # parser.add_argument('--attention_verbose', type=int, default=1, help="attention verbose")

    # 设备配置
    parser.add_argument('--seed', type=int, default=2020, help='random seed')

    # 其它配置
    parser.add_argument('--pretrain', type=bool, default=False, help='whether we use pre-trained weight or not')
    parser.add_argument('--a_hat_folds', type=int, default=100, help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--a_hat_spilit', type=bool, default=False, help="whether we split large adj matrix, like gowalla")
    parser.add_argument('--multicore', type=bool, default=False, help='whether we use multiprocessing or not in test')
    parser.add_argument('--tensorboard', type=bool, default=False, help="enable tensorboard")
    parser.add_argument('--verbose', type=int, default=1, help="verbose")
        
    return parser.parse_args()
