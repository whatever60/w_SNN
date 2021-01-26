from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter


image = Image.open('example.png')
image = image.resize((650,650)).convert('RGB')
x = T.ToTensor()(image)[None]
print(x.shape)
model = torch.load("torch.pkl")
# x.unsqueeze_(0)
writer = SummaryWriter(
        f"/home/tiankang/wusuowei/snn/runs"
    )

writer.add_graph(model, x)
writer.close()
