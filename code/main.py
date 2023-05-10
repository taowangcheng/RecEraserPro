import torch

import utils
import display
import register
from parse import parse_args
from configuration import Config
from models import CL


logo = r"""
██████╗ ███████╗ ██████╗███████╗██████╗  █████╗ ███████╗███████╗██████╗ 
██╔══██╗██╔════╝██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗
██████╔╝█████╗  ██║     █████╗  ██████╔╝███████║███████╗█████╗  ██████╔╝
██╔══██╗██╔══╝  ██║     ██╔══╝  ██╔══██╗██╔══██║╚════██║██╔══╝  ██╔══██╗
██║  ██║███████╗╚██████╗███████╗██║  ██║██║  ██║███████║███████╗██║  ██║
╚═╝  ╚═╝╚══════╝ ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝
"""

if __name__ == '__main__':
    # 加载配置
    args = parse_args()
    config = Config(args)
    display.color_print(logo)
    display.pprint(config(), sort_dicts=False)

    # 初始化设置
    # init tensorboard
    if not config.tensorboard:
        display.color_print("not enable tensorflowboard")
    # 固定随机种子
    utils.set_seed(config.seed)
    print(">>SEED:", config.seed)

    # login
    # 登录模型与数据加载器
    if config.dataloader_name in register.DATALOADERS.keys():
        dataloader = register.DATALOADERS[config.dataloader_name](config)
    else:  # args.dataloader == 'other dataloaders'
        raise NotImplementedError(f"Haven't supported {config.dataloader_name} yet!")
    
    if config.model_name in register.MODELS.keys():
        # if config.model_name == 'RecEraser':
        #     cl_model = CL(config, dataloader)
        #     cl_model = cl_model.to(config.device)
        #     cl_model.train_test(dataloader)
        #     init_users_embeddings = cl_model.init_users_embeddings.weight.detach().cpu().numpy()
        #     init_items_embeddings = cl_model.init_items_embeddings.weight.detach().cpu().numpy()
        #     dataloader.rebuild(init_users_embeddings, init_items_embeddings, config)
        model = register.MODELS[config.model_name](config, dataloader)
    else:  # args.model == 'other models'
        raise NotImplementedError(f"Haven't supported {config.model_name} yet!")

    model = model.to(config.device)  # 把模型的参数放到GPU上
    if config.data_partition:
        for sub_model in model.sub_models:
            sub_model = sub_model.to(config.device)
        # for linear in model.linears:
        #     linear = linear.to(config.device)

    for name, param in model.named_parameters():
        print(name, param.shape)
    # 预训练
    # TODO: 预训练存在问题，抽象出来，RecEraser的Model的参数不包含sub-model的参数
    print(f"load and save to {config.weight_filename}")
    if config.pretrain:
        try:
            model.load_state_dict(torch.load(config.weight_filename, map_location=torch.device('cpu')))
            display.color_print(f"loaded model weights from {config.weight_filename}")
        except FileNotFoundError:
            display.color_print(f"{config.weight_filename} not exists, start from beginning")
    # 训练过程
    try:
        model.train_test(dataloader)
    finally:
        if config.tensorboard_writer:
            config.tensorboard_writer.close()
    
    # # 训练过程
    # try:
    #     if config.data_partition:
    #         procedure.multi_train_test(dataloader, model, config)
    #     else:
    #         procedure.train_test(dataloader, model, config, config.epochs)
    # finally:
    #     if config.tensorboard_writer:
    #         config.tensorboard_writer.close()


'''
base-model: LightGCN
1. 对比学习训练阶段(划分)
参数共享: 所有用户和商品的嵌入
for _ in epochs:
    根据pair的相似性划分一次, 内部迭代若干次(默认50次)
    共享的用户和商品嵌入在每个sub-model中传播一次, 得到最后一层的输出
    根据sub-models的最后输出, 计算loss, 优化模型参数, 用户和商品嵌入是唯一参数
根据最后的用户和商品嵌入, 最后划分一次

2. 子模型训练阶段
保持不变, 与RecEraser一致
3. 聚合训练阶段
保持不变, 与RecEraser一致
'''

'''
RecEraser的tricks总结
1. 在获得注意力网络输出的聚合的用户和商品嵌入后, 再次使用LightGCN的网络结构, 得到最后的用户和商品嵌入, 用于聚合部分的训练和测试
2. 聚合部分的训练中, 对LightGCN输出的最后的用户嵌入使用dropout, 商品嵌入没使用, 测试中禁用dropout
3. 各个sub-model的测试使用的是全局的测试集, 删除训练集中出现的商品时也是使用全局的训练集
'''

'''
early-stopping怎么设置?目前看到的都是Top10的recall 10步也就是100个epoch 没有取得新的最优值
怎么判断过拟合或者判断是否收敛
'''

'''
发现在划分时, 15次迭代后, loss开始增长了, 应用early_stopping来划分?
'''