import os
import sys
import time
import torch
from tensorboardX import SummaryWriter

# # let pandas shut up
# from warnings import simplefilter
# simplefilter(action="ignore", category=FutureWarning)

class Config():
    def __init__(self, args):
        # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 是否必须？

        # 模型配置&&数据集配置
        self.model_name = args.model
        self.dataset_name = args.dataset
        self.dataloader_name = args.dataloader
        
        # 路径配置
        self.root_path = os.path.dirname(os.path.dirname(__file__))
        self.code_path = os.path.join(self.root_path, 'code')
        self.data_path = os.path.join(self.root_path, 'data')
        self.dataset_path = os.path.join(self.data_path, self.dataset_name)
        self.tensorboard_path = os.path.join(self.code_path, 'runs')
        self.weights_path = os.path.join(self.code_path, 'checkpoints')
        sys.path.append(os.path.join(self.code_path, 'sources'))
        # if not os.path.exists(config['tensorboard_path']):
        #     os.makedirs(config['tensorboard_path'], exist_ok=True)
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path, exist_ok=True)
        
        # 超参数配置
        self.epochs = args.epochs
        self.topks = eval(args.topks)
        self.embedding_size = args.embedding_size
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.layers = args.layers
        self.lr = args.lr
        self.decay = args.decay
        self.dropout = args.dropout
        self.keep_prob = args.keep_prob

        # 划分配置
        self.data_partition = args.data_partition
        self.partition_method = args.partition_method
        self.num_shards = args.num_shards
        self.max_iters = args.max_iters
        # self.attention_size = args.attention_size
        self.attention_size = self.embedding_size // 2
        self.attention_epochs = args.attention_epochs
        self.partition_path = os.path.join(self.dataset_path, self.partition_method)
        self.shards_path = os.path.join(self.partition_path, 'num_' + str(self.num_shards))
        if not os.path.exists(self.shards_path):
            os.makedirs(self.shards_path, exist_ok=True)
        # self.attention_verbose = args.attention_verbose
        self.temp = args.temp

        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.seed = args.seed

        # 其它配置
        self.a_hat_split = False
        self.a_hat_folds = args.a_hat_folds
        self.multicore = args.multicore
        
        self.pretrain = args.pretrain
        self.weight_filename = f"{self.model_name}-{self.dataset_name}-{self.layers}-{self.embedding_size}.pth"
        self.weight_filename = os.path.join(self.weights_path, self.weight_filename)
        
        self.tensorboard = args.tensorboard
        if self.tensorboard:
            self.tensorboard_writer = SummaryWriter(os.path.join(self.tensorboard_path, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + self.model_name + "-" + self.dataset_name))
        else:
            self.tensorboard_writer = None
        self.verbose = args.verbose
    
    def __call__(self):
        config_dict = {"模型配置&&数据集配置": {},
                       "路径配置": {},
                       "超参数配置": {},
                       "划分配置": {},
                        "设备配置": {},
                        "其它配置": {}}
        # The following code is written by Github Copilot
        for key, value in self.__dict__.items():
            if key in ['model_name', 'dataset_name', 'dataloader_name']:
                config_dict['模型配置&&数据集配置'][key] = value
            elif key in ['root_path', 'code_path', 'data_path', 'dataset_path', 'tensorboard_path', 'weights_path']:
                config_dict['路径配置'][key] = value
            elif key in ['epochs', 'topks', 'embedding_size', 'train_batch_size', 'test_batch_size', 'layers', 'lr', 'decay', 'dropout', 'keep_prob']:
                config_dict['超参数配置'][key] = value
            elif key in ['data_partition', 'partition_method', 'num_shards', 'max_iters', 'attention_size', 'attention_epochs', 'partition_path', 'shards_path', 'attention_verbose', 'temp']:
                config_dict['划分配置'][key] = value
            elif key in ['device', 'seed']:
                config_dict['设备配置'][key] = value
            elif key in ['a_hat_split', 'a_hat_folds', 'multicore', 'pretrain', 'weight_filename', 'tensorboard', 'tensorboard_writer', 'verbose']:
                config_dict['其它配置'][key] = value
        return config_dict
        
      
