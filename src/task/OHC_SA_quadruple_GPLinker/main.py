import sys
sys.path.append("../../model/GPLinker_quadruple")  # 添加模型的模块
sys.path.append("../../task/OHC_SA_quadruple_GPLinker")  # 添加任务的模块
import os
os.environ["project_root"] = os.path.abspath("../../..")  # 定位到项目根目录，以便后续加载文件
from framework import Framework  # 爆红不用管
from config import Config
from dataLoader import MyDataset, collate_fn
from torch.utils.data import DataLoader
import torch
import numpy as np

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config = Config()
fw = Framework(config)

# 加载数据
train_dataset = MyDataset(config, config.train_data)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, collate_fn=collate_fn,
                              pin_memory=True)
dev_dataset = MyDataset(config, config.dev_data)
dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn, pin_memory=True)
test_dataset = MyDataset(config, config.test_data)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn, pin_memory=True)

fw.train(train_dataloader, dev_dataloader, test_dataloader)
