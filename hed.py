'''https://github.com/sniklaus/pytorch-hed/
'''

from PIL import Image
import torch
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netScoreOne = torch.nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreTwo = torch.nn.Conv2d(
            in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreThr = torch.nn.Conv2d(
            in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreFou = torch.nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreFiv = torch.nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0
        )

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0
            ),
            torch.nn.Sigmoid(),
        )
    # end

    def forward(self, image):
        tenBlue = (image[:, 0:1, :, :] * 255.0) - 104.00698793
        tenGreen = (image[:, 1:2, :, :] * 255.0) - 116.66876762
        tenRed = (image[:, 2:3, :, :] * 255.0) - 122.67891434

        image = torch.cat([tenBlue, tenGreen, tenRed], 1)

        tenVggOne = self.netVggOne(image)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(
            input=tenScoreOne,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        tenScoreTwo = torch.nn.functional.interpolate(
            input=tenScoreTwo,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        tenScoreThr = torch.nn.functional.interpolate(
            input=tenScoreThr,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        tenScoreFou = torch.nn.functional.interpolate(
            input=tenScoreFou,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        tenScoreFiv = torch.nn.functional.interpolate(
            input=tenScoreFiv,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        return self.netCombine(
            torch.cat(
                [tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv], 1
            )
        )
    # end
# end


##########################################################
if __name__ == '__main__':
    image = Image.open('example.png').resize((650, 650)).convert('RGB')
    image = TF.to_tensor(image)
    image.unsqueeze_(0)
    print(image.shape)

    model = Network()
    model.load_state_dict(
            {
                strKey.replace("module", "net"): tenWeight
                for strKey, tenWeight in torch.hub.load_state_dict_from_url(
                    url="http://content.sniklaus.com/github/pytorch-hed/network-bsds500.pytorch",
                    file_name="hed-bsds500",
                ).items()
            }
        )
    model.eval()
    result = model(image)
    plt.imshow(result.detach().numpy()[0][0], 'binary')
    plt.savefig('result.jpg')
