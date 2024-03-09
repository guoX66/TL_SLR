# coding:utf-8
import argparse
import platform
import shutil
import traceback

import yaml
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from _utils.configs import read_cfg, LSTM
from _utils.myutils import *
from common.mico.backbone import MicroNet
from common.mico.utils.defaults import _C as cfg


def pre_model(Train, txt_list, modelinfo):
    gpus = [0, 1]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    model_name = 'model-' + modelinfo.model
    learn_rate, step_size, gamma, divide_present = Train.learn_rate, Train.step_size, Train.gamma, Train.divide_present  # 获取模型参数
    os_name = str(platform.system())
    if os_name == 'Windows':
        num_workers = 0
    else:
        num_workers = 32
    add_log('-' * 40 + '本次训练准备中，准备过程如下' + '-' * 40, txt_list)
    add_log("是否使用GPU训练：{}".format(torch.cuda.is_available()), txt_list)  # 判断是否采用gpu训练

    if torch.cuda.is_available():
        add_log("使用GPU做训练", txt_list)
        add_log("GPU名称为：{}".format(torch.cuda.get_device_name()), txt_list)

    else:
        add_log("使用CPU做训练", txt_list)
    print(f'使用的预训练模型为:{modelinfo.model}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if modelinfo.model != "LSTM":
        LSTM_model = False
        transform = transforms.Compose([
            transforms.Resize([modelinfo.size[0], modelinfo.size[1]]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(modelinfo.ms[0], modelinfo.ms[1])
        ])

        dataset = ImageFolder(Train.imgpath, transform=transform)  # 训练数据集
        class_dict = dataset.class_to_idx
        class_to_id_dict = {class_dict[k]: k for k in class_dict.keys()}

        # torch自带的标准数据集加载函数

    else:
        LSTM_model = True
        _, label_dict, _ = get_label_list(Train.npypath)
        class_to_id_dict = {}
        class_id = 0
        path1 = []
        for i in label_dict.keys():
            class_to_id_dict[i] = class_id
            for j in label_dict[i]:
                tmp_path = os.path.join(Train.npypath, i, j)
                path1.append([tmp_path, class_id])
            class_id += 1

        path1_dict = {i: [] for i in range(len(label_dict))}
        for i in path1:
            path1_dict[i[1]].append(i[0])
        dataset, class_to_id_dict = train_process_np(path1_dict, 20, class_to_id_dict)

    n_label = len(list(class_to_id_dict.keys()))
    write_json(class_to_id_dict, 'class_id')
    train_size = int(Train.divide_present * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    dataloader_val = DataLoader(val_dataset, batch_size=Train.batch_size, shuffle=True, num_workers=num_workers,
                                drop_last=False,
                                pin_memory=True)
    dataloader_train = DataLoader(train_dataset, batch_size=Train.batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=False, pin_memory=True)

    epoch = Train.epoch  # 迭代次数
    if modelinfo.model == "micronet_m3":
        model = MicroNet(cfg, num_classes=n_label)
        model.load_state_dict(torch.load('models/pth/ini-micronet-m3.pth'), strict=False)
    elif modelinfo.model == "LSTM":
        model = LSTM(128, 1, n_label, Train.batch_size, 20, device)
    else:
        model = make_train_model(modelinfo, n_label)
    add_log('-' * 40 + '本次训练准备中，准备过程如下' + '-' * 40, txt_list)
    add_log("是否使用GPU训练：{}".format(torch.cuda.is_available()), txt_list)  # 判断是否采用gpu训练
    if torch.cuda.is_available():
        add_log("使用GPU做训练", txt_list)
        add_log("GPU名称为：{}".format(torch.cuda.get_device_name()), txt_list)

    else:
        add_log("使用CPU做训练", txt_list)

    loss_fn = nn.CrossEntropyLoss().to(device)  # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)  # 可调超参数
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    model = model.to(device)  # 将模型迁移到gpu
    try:
        model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    except AssertionError:
        pass

    return [model_name, model, epoch, dataloader_train, dataloader_val, loss_fn, optimizer, scheduler, txt_list,
            device, LSTM_model]


def train_model(train_args):
    model_name, model, epoch, dataloader_train, dataloader_val, loss_fn, optimizer, scheduler, txt_list, device, LSTM_model = train_args
    loss_list = []
    acc_list = []
    best_acc = 0  # 起步正确率
    ba_epoch = 1  # 标记最高正确率的迭代数
    min_lost = np.Inf  # 起步loss
    ml_epoch = 1  # 标记最低损失的迭代数
    ss_time = time.time()  # 开始时间
    date_time = time.strftime('%Y-%m-%d-%Hh %Mm', time.localtime())
    train_txt = []
    start_time = time.perf_counter()
    for i in range(epoch):
        model.train()
        for j, [imgs, targets] in enumerate(dataloader_train):
            # if j == 1:
            #     os.system("nvidia-smi")
            imgs, targets = imgs.to(device), targets.to(device)
            if LSTM_model:
                targets = targets.view(-1)
            # imgs.float()
            # imgs=imgs.float() #为将上述改为半精度操作，在这里不需要用
            # imgs=imgs.half()
            # print(imgs.shape)
            outputs = model(imgs)
            # print(outputs.shape)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 梯度优化

        scheduler.step()
        model.eval()
        total_val_loss = 0
        total_accuracy = 0
        with torch.no_grad():  # 验证数据集时禁止反向传播优化权重
            n_total_val = 0
            for j, data in enumerate(dataloader_val):
                imgs, targets = data
                # imgs.float()
                # imgs=imgs.float()
                imgs = imgs.to(device)
                targets = targets.to(device)
                if LSTM_model:
                    targets = targets.view(-1)
                n_total_val += len(targets)
                # imgs=imgs.half()
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_val_loss = total_val_loss + loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy

            acc = float(total_accuracy * 100 / n_total_val)
            total_test_loss = float(total_val_loss)
            add_log("验证集上的loss：{:.3f}".format(total_val_loss), train_txt, is_print=False)
            add_log("验证集上的正确率：{:.3f}%".format(acc), train_txt, is_print=False)
            acc_list.append(round(acc, 3))
            loss_list.append(round(total_val_loss, 3))

            if total_accuracy > best_acc:  # 保存迭代次数中最好的模型
                add_log("根据正确率，已修改模型", train_txt, is_print=False)
                best_acc = acc
                ba_epoch = i + 1
                torch.save(model.state_dict(), f'{model_name}.pth')
            if total_val_loss < min_lost:
                add_log("根据损失，已修改模型", train_txt, is_print=False)
                ml_epoch = i + 1
                min_lost = total_test_loss
                torch.save(model.state_dict(), f'{model_name}.pth')

            bar(i + 1, epoch, start_time, '训练进度', train=True, loss=total_test_loss, acc=acc)

    print()
    ee_time = time.time()
    edate_time = time.strftime('%Y-%m-%d-%H时%M分', time.localtime())
    total_time = ee_time - ss_time

    add_log('-' * 35 + f'本次训练结束于{edate_time}，训练结果与报告如下' + '-' * 40, txt_list)
    add_log(f'本次训练开始时间：{date_time}', txt_list)
    add_log("本次训练用时:{}小时:{}分钟:{}秒".format(int(total_time // 3600), int((total_time % 3600) // 60),
                                                     int(total_time % 60)), txt_list)

    add_log(
        f'验证集上在第{ba_epoch}次迭代达到最高正确率，最高的正确率为{round(float(best_acc),3)}%',
        txt_list)
    add_log(f'验证集上在第{ml_epoch}次迭代达到最小损失，最小的损失为{round(float(min_lost),3)}', txt_list)

    return model_name


def train_process():
    from _utils.configs import TrainImg, ModelInfo
    Train = TrainImg()
    modelinfo = ModelInfo()
    txt_list = []
    st = time.time()
    add_log('*' * 40 + f' 训练开始 ' + '*' * 40, txt_list)
    train_args = pre_model(Train, txt_list, modelinfo)
    model_name = train_model(train_args)
    add_log(40 * '*' + f'模型训练参数如下' + 40 * '*', txt_list)
    add_log(f'预训练模型为:{modelinfo.model}', txt_list)
    add_log(
        f'训练迭代数为:{Train.epoch},初始学习率为:{Train.learn_rate},衰减步长和系数为{Train.step_size},{Train.gamma}',
        txt_list)
    add_log(f'数据压缩量为:{Train.batch_size}', txt_list)
    date_time = time.strftime('%Y-%m-%d-%Hh %Mm', time.localtime())
    filename = model_name + '-' + str(date_time)
    model_path = f"log/train_process/{filename}/{filename}"
    train_dir(filename)
    shutil.move(f'{model_name}.pth', f'./{model_path}.pth')
    et = time.time()
    tt_time = et - st
    add_log("训练总共用时:{}小时:{}分钟:{}秒".format(int(tt_time // 3600), int((tt_time % 3600) // 60),
                                                     int(tt_time % 60)), txt_list)
    write_log(f"log/train_process/{filename}", filename, txt_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--divide_present', type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--step_size', type=int)
    parser.add_argument('--gamma', type=float)
    args = parser.parse_args()
    dic = vars(args)
    change_dict = {i: dic[i] for i in dic.keys() if dic[i] is not None}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tr_Cfg, Ir_Cfg, ba_Cfg, Cfg = read_cfg(base_dir)
    for i in change_dict.keys():
        Cfg['train'][i] = change_dict[i]
    with open('Cfg.yaml', "w", encoding="utf-8") as f:
        yaml.dump(Cfg, f)
    try:
        train_process()
    except Exception as e:
        print(e)
        try:
            os.mkdir('log')
        except:
            pass
        traceback.print_exc(file=open('log/train_err_log.txt', 'w+'))
