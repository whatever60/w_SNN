import os
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader

from tqdm.auto import tqdm

import numpy as np

import spiking_vgg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def find_threshold(model, train_loader, DEVICE):

    data, _ = next(iter(train_loader))
    data = transform_train(data.to(DEVICE))
    thresholds = []
    for idx, layer in tqdm(chain(model.features.named_children(), model.classifier.named_children())):  # `named_children`：子模块的迭代器
        idx = int(idx)
        if idx + 1 == len(model.features) + len(model.classifier):
            # print(1)
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # for each layer, do this for only one batch then break.
            with torch.no_grad():
                model.eval()
                output = model(data, find_max_mem=True, max_mem_layer=idx)
                thresholds.append(output.item())
    return thresholds


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "9"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print(DEVICE)
    SEED = 42
    # Seed random number
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    TRAIN = True
    DATASET = 'CIFAR10'
    DATA_ROOT = '/home/tiankang/wusuowei/dataset'
    BATCH_SIZE = 32
    ARCH = 'VGG16'
    LR = 0.0001
    pretrained_ann = torch.load('./ann_vgg16_cifar10.pth')
    EPOCH = 300
    LR_INTERVAL = [180, 240, 270]
    LR_REDUCE = 10
    TIMESTEP = 100
    LEAK = 1.0
    SCALING_FACTOR = 0.7
    DEFAULT_THRESHOLD = 1.0
    ACTIVATION = 'STDB'
    ALPHA = 0.3
    BETA = 0.01
    OPTIMIZER = 'Adam'
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.95
    AMSGRAD = True
    DROPOUT = 0.3
    KERNEL_SIZE = 3
    # WORKER_NUM = 4

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transform_train = torch.nn.Sequential(
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        normalize,
    )
    transform_test = normalize

    trainset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    testset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    labels = 10

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)#, num_workers=WORKER_NUM)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)#, num_workers=WORKER_NUM)

    model = spiking_vgg.VGG_SNN_STDB(
        vgg_name=ARCH,
        activation=ACTIVATION,
        labels=labels,  # to determine number of neuron in the output layer
        timesteps=TIMESTEP,
        leak=LEAK,
        default_threshold=DEFAULT_THRESHOLD,
        alpha=ALPHA,
        beta=BETA,
        dropout=DROPOUT,
        kernel_size=KERNEL_SIZE,
        dataset=DATASET,
    )
    model.to(DEVICE)
    # model = nn.DataParallel(model)

    if pretrained_ann is not None:
        state = pretrained_ann['state_dict'] if 'state_dict' in pretrained_ann else pretrained_ann
        del state['thresholds']
        cur_dict = model.state_dict()
        for k, v in state.items():
            if k in cur_dict and v.shape == cur_dict[k].shape:
                cur_dict[k] = nn.Parameter(v.data)
                print('Converted')
        model.load_state_dict(cur_dict)
        if 'thresholds' in state:
            thresholds = state['thresholds']
            model.threshold_update(scaling_factor=SCALING_FACTOR, thresholds=thresholds)
        else:
            print('Begin looking for threshold')
            thresholds = find_threshold(model, train_loader, DEVICE)
            model.threshold_update(scaling_factor=SCALING_FACTOR, thresholds=thresholds)
            temp = {}
            for key,value in state.items():
                temp[key] = value
            temp['thresholds'] = thresholds
            torch.save(temp, './ann_vgg16_cifar10.pth')  # state_dict
            print('Threshold searching over..')

    if OPTIMIZER == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=LR,
            amsgrad=AMSGRAD,
            weight_decay=WEIGHT_DECAY,
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            momentum=MOMENTUM,
        )
    criterion = nn.CrossEntropyLoss()

    max_accuracy = 0
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    loop1 = tqdm(range(EPOCH), leave=False)
    for epoch in loop1:

        if TRAIN:
            model.network_update(timesteps=TIMESTEP, leak=LEAK)
            losses = AverageMeter("Loss")
            top1 = AverageMeter("Acc@1")

            if epoch in LR_INTERVAL:
                for param_group in optimizer.param_groups:
                    param_group["lr"] /= LR_REDUCE

            model.train()

            loop2 = tqdm(train_loader, leave=False)
            for batch_idx, (data, target) in enumerate(loop2):
                data, target = transform_train(data.to(DEVICE)), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                pred = output.max(1, keepdim=True)[1]
                correct = pred.eq(target.data.view_as(pred)).cpu().sum()

                losses.update(loss.item(), data.size(0))
                top1.update(correct.item() / data.size(0), data.size(0))
                loss, acc = losses.avg, top1.avg
                # print(batch_idx, loss, acc)
                loop2.set_postfix(dict(loss=loss, acc=acc))
                break
            loss, acc = losses.avg, top1.avg
            loop1.set_postfix(dict(loss=loss, acc=acc))
            train_loss.append(loss)
            train_acc.append(acc)
        continue
        losses = AverageMeter("Loss")
        top1 = AverageMeter("Acc@1")
        with torch.no_grad():
            model.eval()
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = transform_test(data.to(DEVICE)), target.to(DEVICE)
                output = model(data)
                loss = criterion(output, target)
                pred = output.max(1, keepdim=True)[1]
                correct = pred.eq(target.data.view_as(pred)).cpu().sum()

                losses.update(loss.item(), data.size(0))
                top1.update(correct.item() / data.size(0), data.size(0))

            temp1 = []
            for value in model.threshold.values():
                temp1 = temp1 + [value.item()]

            if epoch > 5 and top1.avg < 0.15:
                print("\n Quitting as the training is not progressing")
                exit(0)

            if top1.avg > max_accuracy:
                max_accuracy = top1.avg

                state = {
                    "accuracy": max_accuracy,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "thresholds": temp1,
                    "timesteps": TIMESTEP,
                    "leak": LEAK,
                    "activation": ACTIVATION,
                }
                filename = "./trained_models/snn/" + 'snn' + ".pth"
                torch.save(state, filename)
            loss, acc = losses.avg, top1.avg
            loop1.set_postfix(dict(test_loss=loss, test_acc=acc))
            test_loss.append(loss)
            test_acc.append(acc)        
