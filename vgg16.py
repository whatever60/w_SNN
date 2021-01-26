'''https://github.com/chengyangfu/pytorch-vgg-cifar10
'''

import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as T
import torchvision.datasets as datasets


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    for i, (image, target) in enumerate(train_loader):
        # measure data loading time
        image = image.to(device)
        target = target.cuda()
        if HALF:
            image = image.half()
        # compute output
        output = model(image)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    for i, (image, target) in enumerate(val_loader):
        image = image.to(device)
        target = target.to(device)

        if HALF:
            image = image.half()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        # measure elapsed time
    #     if i % args.print_freq == 0:
    #         print('Test: [{0}/{1}]\t'
    #               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
    #                   i, len(val_loader), batch_time=batch_time, loss=losses,
    #                   top1=top1))

    # print(' * Prec@1 {top1.avg:.3f}'
    #       .format(top1=top1))
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = LR * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':

    CPU = False
    RESUME = None
    START_EPOCH = 0
    EPOCH = 300
    IMG_ROOT = '/home/tiankang/wusuowei/dataset'
    BATCH_SIZE = 256
    WORKER_NUM = 4
    HALF = True
    EVALUATE = False
    LR = 0.05
    MOMENTUM = 0.5
    WEIGHT_DECAY = 0.01
    SAVE_DIR = './VGG16_CHECKPOINT'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = VGG(make_layers(cfg['vgg16']))
    model.features = torch.nn.DataParallel(model.features)
    model.to(device)

    # optionally resume from a checkpoint
    if RESUME is not None:
        checkpoint = torch.load(RESUME)
        START_EPOCH = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])

    cudnn.benchmark = True

    train_transforms = torch.nn.Sequential(
        T.ToTensor().to(device),
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, 4),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    test_transforms = torch.nn.Sequenctial(
        T.ToTensor().to(device),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    train_transforms = torch.jit.script(train_transforms)
    test_transforms = torch.jir.script(test_transforms)

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=IMG_ROOT,
            train=True,
            transform=train_transforms,
            download=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKER_NUM,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=IMG_ROOT, train=False, transform=test_transforms),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKER_NUM,
        pin_memory=True
    )

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss()

    if HALF:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(
        model.parameters(),
        LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    if EVALUATE:
        validate(test_loader, model, criterion)

    best_prec1 = 0
    for epoch in range(START_EPOCH, EPOCH):
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        prec1 = validate(test_loader, model, criterion)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        
        if not (epoch + 1) % 30:
            save_checkpoint(
                dict(
                    epoch=epoch + 1,
                    state_dict=model.state_dict(),
                    best_prec1=best_prec1,
                ),
                is_best, 
                filename=os.path.join(SAVE_DIR, f'checkpoint_{epoch}.tar')
            )

