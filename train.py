# coding:utf-8
import platform
import traceback
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from _utils.myutils import *
from _utils.configs import TrainImg, ModelInfo
from torchvision import transforms
import shutil
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from common.mico.backbone import MicroNet
from common.mico.utils.defaults import _C as cfg



def train_model(Train, txt_list, modelinfo):
    gpus = [0, 1]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))

    model_name = 'model-' + modelinfo.model
    transform = transforms.Compose([
        transforms.Resize([modelinfo.size[0], modelinfo.size[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(modelinfo.ms[0], modelinfo.ms[1])
    ])
    learn_rate, step_size, gamma, divide_present = Train.learn_rate, Train.step_size, Train.gamma, Train.divide_present  # 获取模型参数
    loss_list = []
    acc_list = []
    dataset_train = ImageFolder(Train.imgpath, transform=transform)  # 训练数据集
    class_to_id_dict = dataset_train.class_to_idx
    class_dict = {class_to_id_dict[k]: k for k in class_to_id_dict.keys()}
    write_json(class_dict, 'class_id')
    train_size = int(divide_present * len(dataset_train))
    val_size = len(dataset_train) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset_train, [train_size, val_size])
    # torch自带的标准数据集加载函数
    os_name = str(platform.system())
    if os_name == 'Windows':
        num_workers = 0
    else:
        num_workers = 32
    dataloader_val = DataLoader(val_dataset, batch_size=Train.batch_size, shuffle=True, num_workers=num_workers,
                                drop_last=False,
                                pin_memory=True)
    dataloader_train = DataLoader(train_dataset, batch_size=Train.batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=False, pin_memory=True)

    add_log('-' * 40 + '本次训练准备中，准备过程如下' + '-' * 40, txt_list)
    add_log("是否使用GPU训练：{}".format(torch.cuda.is_available()), txt_list)  # 判断是否采用gpu训练

    if torch.cuda.is_available():
        add_log("使用GPU做训练", txt_list)
        add_log("GPU名称为：{}".format(torch.cuda.get_device_name()), txt_list)

    else:
        add_log("使用CPU做训练", txt_list)
    print(f'使用的预训练模型为:{modelinfo.model}')

    n_label = len(list(class_to_id_dict.keys()))
    if modelinfo.model == "micronet_m3":
        model = MicroNet(cfg, num_classes=n_label)
        model.load_state_dict(torch.load('mico/models/micronet-m3.pth'), strict=False)
    else:
        model = make_train_model(modelinfo, n_label)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.CrossEntropyLoss().to(device)  # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)  # 可调超参数
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    epoch = Train.epoch  # 迭代次数
    best_acc = 0  # 起步正确率
    ba_epoch = 1  # 标记最高正确率的迭代数
    min_lost = np.Inf  # 起步loss
    ml_epoch = 1  # 标记最低损失的迭代数
    model = model.to(device)  # 将模型迁移到gpu
    try:
        model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    except AssertionError:
        pass

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
                best_acc = total_accuracy
                ba_epoch = i + 1
                torch.save(model.state_dict(), f'{model_name}.pth')
            if total_val_loss < min_lost:
                add_log("根据损失，已修改模型", train_txt, is_print=False)
                ml_epoch = i + 1
                min_lost = total_val_loss
                torch.save(model.state_dict(), f'{model_name}.pth')

            bar(i + 1, epoch, start_time, '训练进度', train=True, loss=total_test_loss, acc=acc)

    print()
    if Train.write_process:
        txt_list = txt_list + train_txt
    ee_time = time.time()
    edate_time = time.strftime('%Y-%m-%d-%H时%M分', time.localtime())
    total_time = ee_time - ss_time

    add_log('-' * 35 + f'本次训练结束于{edate_time}，训练结果与报告如下' + '-' * 40, txt_list)
    add_log(f'本次训练开始时间：{date_time}', txt_list)
    add_log("本次训练用时:{}小时:{}分钟:{}秒".format(int(total_time // 3600), int((total_time % 3600) // 60),
                                                     int(total_time % 60)), txt_list)

    add_log(
        f'训练集图片数为:{len(train_dataset)},验证集图片数为:{len(val_dataset)},拆分比为{int(divide_present * 10)}:{int(10 - divide_present * 10)}',
        txt_list)

    add_log(
        f'验证集上在第{ba_epoch}次迭代达到最高正确率，最高的正确率为{round(float(best_acc * 100 / len(val_dataset)), 3)}%',
        txt_list)
    add_log(f'验证集上在第{ml_epoch}次迭代达到最小损失，最小的损失为{round(min_lost, 3)}', txt_list)
    add_log('-' * 40 + '下面开始在测试集上测试' + '-' * 40, txt_list)
    return model_name, loss_list, acc_list, epoch, acc


def train_process():
    Train = TrainImg()
    modelinfo = ModelInfo()
    txt_list = []
    st = time.time()
    num = 1
    add_log('*' * 40 + f' 训练开始 ' + '*' * 40, txt_list)
    add_log('*' * 60 + f"第{num}次训练开始" + '*' * 60, txt_list)
    model_name, loss_list, acc_list, epoch, acc = train_model(Train, txt_list, modelinfo)
    num += 1

    add_log(40 * '*' + f'模型训练参数如下' + 40 * '*', txt_list)
    add_log(f'预训练模型为:{modelinfo.model}', txt_list)
    add_log(
        f'训练迭代数为:{epoch},初始学习率为:{Train.learn_rate},衰减步长和系数为{Train.step_size},{Train.gamma}',
        txt_list)
    add_log(f'数据压缩量为:{Train.batch_size}', txt_list)
    add_log(f'图片预处理大小为:{modelinfo.size[0]}x{modelinfo.size[1]}', txt_list)
    add_log(f'图片标准化设置为:{modelinfo.ms[0]}和{modelinfo.ms[1]}', txt_list)
    date_time = time.strftime('%Y-%m-%d-%Hh %Mm', time.localtime())
    filename = model_name + '-' + str(date_time)
    model_path = f"train_process/{filename}/{filename}"
    train_dir(filename)
    shutil.move(f'{model_name}.pth', f'./{model_path}.pth')
    et = time.time()
    tt_time = et - st
    add_log("训练总共用时:{}小时:{}分钟:{}秒".format(int(tt_time // 3600), int((tt_time % 3600) // 60),
                                                     int(tt_time % 60)), txt_list)
    write_log(f"train_process/{filename}", filename, txt_list)
    make_plot(loss_list, 'loss', filename, epoch)
    make_plot(acc_list, 'acc', filename, epoch)


if __name__ == '__main__':
    try:
        train_process()
    except Exception as e:
        print(e)
        try:
            os.mkdir('log')
        except:
            pass
        traceback.print_exc(file=open('log/train_err_log.txt', 'w+'))
