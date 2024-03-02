import torchvision.models as models
from torchvision import transforms
import os
import yaml

with open('Cfg.yaml', 'r', encoding='utf-8') as f:
    Cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
tr_Cfg = Cfg['train']
Ir_Cfg = Cfg['inference']


class ModelInfo:
    def __init__(self):
        self.model = tr_Cfg['model']  # 选择模型，可选googlenet,resnet18，resnet34，resnet50，resnet101
        self.modelname = 'model-' + self.model
        self.size = [int(tr_Cfg['size'].split('x')[0]), int(tr_Cfg['size'].split('x')[1])]  # 设置输入模型的图片大小
        # self.ms = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]  # 标准化设置
        self.ms = tr_Cfg['Normalized_matrix']
        self.min_pr = 4 / 5  # 设置预测时各模型预测结果最多的类作为最终结果所需要的最小占比，即几票通过算通过


class TrainImg:
    def __init__(self):
        self.imgpath = 'data/images'  # 保存图片的文件名
        self.foldname = 'data/static'
        self.divide_present = tr_Cfg['divide_present']
        self.batch_size = tr_Cfg['batch_size']
        self.learn_rate = tr_Cfg['learn_rate']
        self.step_size = tr_Cfg['step_size']
        self.gamma = tr_Cfg['gamma']
        self.epoch = tr_Cfg['epoch']
        self.show_mode = tr_Cfg['show_mode']
        self.write_process = tr_Cfg['write_process']


class TestImg(TrainImg):
    def __init__(self):
        super().__init__()
        self.foldname = 'static/test_wav'
        self.imgpath = 'static/test_img'  # 保存测试图片的路径名
        self.log_path = 'log-test'
