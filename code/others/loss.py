from torch import optim

from models import BasicModel
from configuration import Config

class BPRLoss:
    def __init__(self, model: BasicModel, config: Config):
        self.decay = config.decay
        self.lr = config.lr
        self.model = model
        # TODO: model.parameters()不包含线性层的参数
        # 不在外面，无法优化线性层的参数
        # 同一个tensor中其它sub-model初始嵌入受影响吗
        # 那抽样训练是不是只优化抽到的人和商品
        self.opt = optim.Adam(model.parameters(), lr=self.lr) 

    def step(self, users, pos_items, neg_items):
        loss, reg_loss = self.model.bpr_loss(users, pos_items, neg_items)
        reg_loss = reg_loss * self.decay
        loss = loss + reg_loss

        # 优化模型参数
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()  # 返回loss的值