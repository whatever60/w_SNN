import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class PoissonGenerator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image):

        out = torch.mul(
            torch.le(torch.rand_like(image), torch.abs(image) * 1.0).float(),
            torch.sign(image),
        )
        return out


class STDB(torch.autograd.Function):

    alpha = ""
    beta = ""

    @staticmethod
    def forward(ctx, image, last_spike):

        ctx.save_for_backward(last_spike)
        out = torch.zeros_like(image).cuda()
        out[image > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):

        (last_spike,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = STDB.alpha * torch.exp(-1 * last_spike) ** STDB.beta
        return grad * grad_input, None


class HED_SNN_STDB(nn.Module):
    def __init__(
        self,
        activation=STDB,
        timesteps=100,
        leak=1.0,
        default_threshold=1.0,
        alpha=0.3,
        beta=0.01,
        # dropout=0.2,  I think this hasn't been invented at HED's time
    ):
        super().__init__()
        self.act_func = activation.apply
        self.timesteps = timesteps
        self.leak = torch.tensor(leak)
        STDB.alpha = alpha
        STDB.beta = beta
        # self.dropout = dropout
        self.input_layer = PoissonGenerator()
        self.threshold = {}
        self.mem = {}
        # self.mask = {}
        self.spike = {}

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.ReLU(inplace=False),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.ReLU(inplace=False),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.ReLU(inplace=False),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.ReLU(inplace=False),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.ReLU(inplace=False),
        )

        self.scores = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.Conv2d(
                in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.Conv2d(
                in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.Conv2d(
                in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.Conv2d(
                in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False
            )
        )

        self.netCombine = nn.Conv2d(
                in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True
            )
        self._initialize_weights2()
        for l in range(len(self.features)):
            if isinstance(self.features[l], nn.Conv2d):
                self.threshold[l] = torch.tensor(default_threshold)

    def forward(self, image, find_max_mem=False, max_mem_layer=0):
        self.neuron_init(image)
        max_mem = 0.0
        output_layers = [3, 8, 15, 22, 29]
        outputs = []

        for t in range(self.timesteps):
            out_prev = self.input_layer(image)

            for idx, feature in enumerate(self.features):
                if isinstance(feature, nn.Conv2d):
                    if find_max_mem and idx == max_mem_layer:
                        max_mem = max(feature(out_prev).max(), 0.0)
                        break
                    mem_thr = self.mem[idx] / self.threshold[idx] - 1  # same as self.mem - self.threshold?
                    out = self.act_func(mem_thr, t - 1 - self.spike[idx])
                    rst = self.threshold[idx] * (mem_thr > 0).float()  # same as threshold * out?
                    self.spike[idx] = self.spike[idx].masked_fill(out.bool(), t - 1)
                    self.mem[idx] = self.leak * self.mem[idx] + feature(out_prev) - rst
                    out_prev = out.clone()
                else:
                    out_prev = feature(out_prev)
                # elif isinstance(feature, nn.Dropout):
                #     out_prev = out_prev * self.mask[idx]
                if idx in output_layers:
                    output_idx = output_layers.index(idx)
                    if output_idx == len(outputs):  # or t == 0.
                        outputs.append(self.scores[output_idx](out_prev))
                    else:
                        outputs[output_idx] += self.scores[output_idx](out_prev)
        else:
            if find_max_mem:
                return max_mem
            else:
                outputs = [self.interpolate(f, image) for f in outputs]
                outputs.append(self.netCombine(torch.cat(outputs, 1)))
                return outputs        
        
    def _initialize_weights2(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def threshold_update(self, thresholds, scaling_factor=1.0):
        # Initialize thresholds
        self.scaling_factor = scaling_factor

        for pos in range(len(self.features)):
            if isinstance(self.features[pos], nn.Conv2d):
                if thresholds:
                    self.threshold[pos] = torch.tensor(thresholds.pop(0)*self.scaling_factor)
		
    def neuron_init(self, x):
        self.batch_size = x.size(0)
        self.width = x.size(2)
        self.height = x.size(3)

        for l in range(len(self.features)):
            if isinstance(self.features[l], nn.Conv2d):
                self.mem[l] = torch.zeros(
                    self.batch_size,
                    self.features[l].out_channels,
                    self.width,
                    self.height,
                )
            # elif isinstance(self.features[l], nn.Dropout):
            #     self.mask[l] = self.features[l](torch.ones(self.mem[l - 2].shape))
            elif isinstance(self.features[l], nn.MaxPool2d):
                self.width = self.width // self.features[l].kernel_size
                self.height = self.height // self.features[l].kernel_size

        self.spike = copy.deepcopy(self.mem)
        for key, values in self.spike.items():
            for value in values:
                value.fill_(-1000)

    def network_update(self, timesteps, leak):
        self.timesteps = timesteps
        self.leak = torch.tensor(leak)

    @staticmethod
    def interpolate(feature, image):
        return F.interpolate(
            input=feature,
            size=image.shape[2:4],
            mode='bilinear',
            align_corners=False
        )
    
