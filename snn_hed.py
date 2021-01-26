import os
import random

import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.optim as optim
from PIL import Image
from tqdm.auto import tqdm
import numpy as np

import spiking_hed


class AverageMeter():
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



class BSD500DatasetTrain:
    def __init__(self, ROOT):

        self.rootdir = ROOT

        ### data 
        self.all_path_list = [i for i in os.listdir('/'.join([ROOT, 'images', 'train'])) if i.endswith('jpg')]
        # with open('/'.join([self.rootdir, self.train_list]), 'r') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         line = line[:-1]
        #         cur_pair = line.split(' ')
                
        #         self.all_path_list.append( cur_pair )
        # print('in data_loader: Train data preparation done')

    def mytransfrom(self, img, gt):
        '''
        input:  img,gt, PIL image
        output: tensor
        '''
        ### ToTensor
        img = TF.to_tensor(img)
        gt = TF.to_tensor(gt)
        
        ### ColorJitterUG:
        if ColorJitter:
            color_jitter = transforms.ColorJitter(
                brightness = brightness,
                contrast = contrast,
                saturation = saturation,
                hue = hue
            )
            color_jitter_transform = color_jitter.get_params(
                color_jitter.brightness,
                color_jitter.contrast,
                color_jitter.saturation,
                color_jitter.hue
            )
            img = color_jitter_transform(img)
        
        if HFlip:
            if random.random() > 0.5:
                img = TF.hflip(img)
                gt = TF.hflip(gt)

        ### Normalization
        mean = [float(item) / 255.0 for item in MEAN]
        std = [1,1,1]

        normalizer = transforms.Normalize(mean=mean, std=std)
        img = normalizer(img)
    
        return img, gt
        

    def __getitem__(self, idx):
        img_path = '/'.join([self.rootdir, 'images', 'train', self.all_path_list[idx]])
        gt_path = '/'.join([self.rootdir, 'groundTruth', 'train', self.all_path_list[idx]])[:-3] + 'mat'

        img = Image.open(img_path).convert('RGB')  # 481 * 321
        gt  = Image.fromarray(scipy.io.loadmat(gt_path)['groundTruth'][0][0][0][0][1])
        if img.size != (481, 321):
            img = img.rotate(90, expand=True)
            gt = gt.rotate(90, expand=True)
        assert img.size == (481, 321), (img.size, img_path)

        img = img.resize(SHAPE)
        gt = gt.resize(SHAPE)

        img_t, gt_t = self.mytransfrom(img, gt)

        if gt_mode =='gt_half':
            gt_t[gt_t>=0.5] = 1 
            gt_t[gt_t<0.5] = 0
        
        return img_t, gt_t

    def __len__(self):
        return len(self.all_path_list)


class BSD500DatasetTest:
    def __init__(self, ROOT):
        self.rootdir = ROOT
        self.all_path_list = [i for i in os.listdir('/'.join([ROOT, 'images', 'test'])) if i.endswith('jpg')]
        
        ### data 
        # self.all_path_list = []
        # with open('/'.join([self.rootdir, self.train_list]), 'r') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         line = line[:-1]
        #         self.all_path_list.append( line )
        # print('in data_loader: Test data preparation done')

        ### transformer
        mean = [float(item) / 255.0 for item in MEAN]
        std = [1,1,1]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
        

    def __getitem__(self, idx):
        img_path = '/'.join([self.rootdir, 'images', 'test', self.all_path_list[idx]])
        gt_path = '/'.join([self.rootdir, 'groundTruth', 'test', self.all_path_list[idx]])[:-3] + 'mat'

        img = Image.open(img_path).convert('RGB')  # 321 * 481
        gt  = Image.fromarray(scipy.io.loadmat(gt_path)['groundTruth'][0][0][0][0][1])
        
        gt_t = TF.to_tensor(gt)
        img_t = self.transform(img)
        
        return img_t, gt_t

    
    def __len__(self):
        return len(self.all_path_list)


def find_threshold(model, dataloader, DEVICE):

    data, _ = next(iter(dataloader))
    data = data.to(DEVICE)

    thresholds = []
    for idx, layer in tqdm(model.features.named_children()):  # `named_children`：子模块的迭代器
        idx = int(idx)

        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # for each layer, do this for only one batch then break.
            with torch.no_grad():
                model.eval()
                output = model(data, find_max_mem=True, max_mem_layer=idx)
                thresholds.append(output.item())
    return thresholds


def edge_weight(target):

        h, w = target.shape[2:]
        #num_nonzero = torch.nonzero(target).shape[0]

        #weight_p = num_nonzero / (h*w)
        weight_p = torch.sum(target) / (h * w)
        weight_n = 1 - weight_p

        res = target.clone()
        res[target==0] = weight_p
        res[target>0] = weight_n
        assert (weight_p + weight_n)==1, "weight_p + weight_n !=1"
        #print(res, type(res))
    
        return res


def set_random(SEED):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "9"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(DEVICE)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
    SEED = 42
    set_random(SEED)
    ROOT = '/home/tiankang/wusuowei/dataset/BSDS500/BSDS500/data'
    # ROOT = '/content/BSDS500/BSR/BSDS500/data'  # where is your data

    EPOCH = 10
    BATCH_SIZE = 4
    WORKER_NUM = 1  # dataloader
    SHAPE = (100, 100)  # reshape the image to reduce DRAM usage
    TRAIN = True  # if you just want to do test, set this to False
    # HALF = False  # use half precision
    LOSS_FUNC_LOGITS = True  # which loss function to use
    LOSS_REDUCE = True
    LOSS_WEIGHT_LIST = [1,1,1,1,1,1]
    ANN_PATH = './ann_hed_bsds500.pth'  # where is your pretrained ANN model
    pretrained_ann = torch.load(ANN_PATH, map_location='cpu')
    
    # Params specific for SNN
    TIMESTEP = 100
    LEAK = 1.0  # Membrane potential leak after each timestep
    SCALING_FACTOR = 0.7  # used when estimating threshold using ANN-converted SNN
    DEFAULT_THRESHOLD = 1.0  # threshold membrane potential of firing a spike
    ALPHA = 0.3
    BETA = 0.01  # for STDB backprop

    # Optimizer params
    OPTIMIZER = 'Adam'  # Adam or SGD
    LR = 0.0001  # for both
    LR_INTERVAL = [180, 240, 270]  # LR decay happens at these epochs.
    LR_REDUCE = 10  # decay to 1/LR_REDUCE
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.95  # for SGD
    AMSGRAD = True  # for Adam 

    # Params for image preprocessing and augmentation
    MEAN = [122.67892, 116.66877, 104.00699]
    ColorJitter = True
    brightness = 0.3
    contrast = 0.3
    saturation = 0.3
    hue = 0.1
    HFlip = True
    gt_mode = None

    train_loader = torch.utils.data.DataLoader(
            BSD500DatasetTrain(ROOT),
            batch_size=BATCH_SIZE,
            shuffle=False,  # for overfitting test
            num_workers=WORKER_NUM,
        )

    test_loader = torch.utils.data.DataLoader(
            BSD500DatasetTest(ROOT), 
            batch_size=1,
            shuffle=False,
            num_workers=WORKER_NUM
        )
    
    model = spiking_hed.HED_SNN_STDB(
        timesteps=TIMESTEP,
        leak=LEAK,
        default_threshold=DEFAULT_THRESHOLD,
        alpha=ALPHA,
        beta=BETA,
    )
    # model = nn.DataParallel(model)

    model.to(DEVICE)

    if pretrained_ann is not None:
        state = pretrained_ann['state_dict'] if 'state_dict' in pretrained_ann else pretrained_ann
        # del state['thresholds']
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
            torch.save(temp, ANN_PATH)  # state_dict
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
    
    criterion = F.binary_cross_entropy_with_logits if LOSS_FUNC_LOGITS else F.binary_cross_entropy

    max_accuracy = 0
    train_loss = []
    test_loss = []
    loop1 = tqdm(range(EPOCH), leave=False)
    for epoch in loop1:

        if TRAIN:
            model.network_update(timesteps=TIMESTEP, leak=LEAK)
            loss_meter = AverageMeter("Loss")

            if epoch in LR_INTERVAL:
                for param_group in optimizer.param_groups:
                    param_group["lr"] /= LR_REDUCE

            model.train()

            loop2 = tqdm(train_loader, leave=False)
            for batch_idx, (data, target) in enumerate(loop2):
                data, target = data.to(DEVICE), target.to(DEVICE)
                outputs = model(data)

                # =========== calculate loss =============
                if not LOSS_FUNC_LOGITS:
                    outputs = list(map(nn.Sigmoid(), outputs))
                cur_weight = edge_weight(target)
                losses = [criterion(o, target.float(), weight=cur_weight, reduce=LOSS_REDUCE) for o in outputs]
                loss = sum(l * w for l, w in zip(losses, LOSS_WEIGHT_LIST))
                loss.backward()
                optimizer.zero_grad()
                optimizer.step()
                # =========================================

                loss_meter.update(loss.item(), data.size(0))
                loss = loss_meter.avg
                # print(batch_idx, loss, acc)
                loop2.set_postfix(dict(loss=loss))
                # break  # Let's try overfitting a single batch
            loss = loss_meter.avg
            loop1.set_postfix(dict(loss=loss))
            train_loss.append(loss)
        # continue  # Let's try overfitting a single batch

        loss_meter = AverageMeter("Loss")
        with torch.no_grad():
            model.eval()
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                outputs = model(data)

                # =========== calculate loss =============
                if not LOSS_FUNC_LOGITS:
                    outputs = list(map(nn.Sigmoid(), outputs))
                cur_weight = edge_weight(target)
                losses = [criterion(o, target.float(), weight=cur_weight, reduce=LOSS_REDUCE) for o in outputs]
                loss = sum(l * w for l, w in zip(losses, LOSS_WEIGHT_LIST))
                # =========================================

                loss_meter.update(loss.item(), data.size(0))

            temp1 = []
            for value in model.threshold.values():
                temp1 = temp1 + [value.item()]

            # if epoch > 5 and top1.avg < 0.15:
            #     print("\n Quitting as the training is not progressing")
            #     exit(0)

            # if top1.avg > max_accuracy:
            #     max_accuracy = top1.avg

            #     state = {
            #         "accuracy": max_accuracy,
            #         "epoch": epoch,
            #         "state_dict": model.state_dict(),
            #         "optimizer": optimizer.state_dict(),
            #         "thresholds": temp1,
            #         "timesteps": TIMESTEP,
            #         "leak": LEAK,
            #         "activation": ACTIVATION,
            #     }
            #     filename = "./trained_models/snn/" + 'snn' + ".pth"
            #     torch.save(state, filename)
            loss = loss_meter.avg
            loop1.set_postfix(dict(test_loss=loss))
            test_loss.append(loss)

