import torch
import torchvision.models as models
from torch import nn
from torchvision import transforms
import os
import yaml





def read_cfg(base_dir):
    path = os.path.join(base_dir, 'Cfg.yaml')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            Cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        tr_Cfg = Cfg['train']
        Ir_Cfg = Cfg['inference']
        ba_Cfg = Cfg['base']
        return tr_Cfg, Ir_Cfg, ba_Cfg, Cfg
    else:
        raise FileNotFoundError('Cfg.yaml not found')


class ModelInfo:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(base_dir)
        tr_Cfg, Ir_Cfg, ba_Cfg, Cfg = read_cfg(base_dir)
        self.model = tr_Cfg['model']  # 选择模型，可选googlenet,resnet18，resnet34，resnet50，resnet101
        self.modelname = 'model-' + self.model
        self.size = [256, 256]  # 输入模型的图片大小
        # self.ms = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]  # 标准化设置
        self.ms = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


class TrainImg:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(base_dir)
        tr_Cfg, Ir_Cfg, ba_Cfg, Cfg = read_cfg(base_dir)
        self.imgpath = 'data/images'  # 保存图片的文件名
        self.foldname = 'data/static'
        self.npypath = 'data/npys'
        self.divide_present = tr_Cfg['divide_present']
        self.batch_size = tr_Cfg['batch_size']
        self.learn_rate = tr_Cfg['lr']
        self.step_size = tr_Cfg['step_size']
        self.gamma = tr_Cfg['gamma']
        self.epoch = tr_Cfg['epoch']



class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, output_size, batch_size, high_pot, device):
        super().__init__()
        self.device = device
        self.input_size = 48 * high_pot
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size
        self.high_pot = high_pot
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(2, self.high_pot)
        self.pc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        high_pot_tensor = self.linear(input_seq)
        input_tensor = high_pot_tensor.view(batch_size, seq_len, -1)
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_tensor, (h_0, c_0))  # output(5, 30, 64)
        pred = self.pc(output)  # (5, 30, 1)
        pred = pred[:, -1, :]  # (5, 1)
        return pred
