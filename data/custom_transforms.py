import torch
from torchvision.transforms import v2

class CenterCropSquare(torch.nn.Module):
    def forward(self, img):
        min_side = min(img.width, img.height)
        top = max(0, (img.height - min_side) // 2)
        left = max(0, (img.width - min_side) // 2)
        img = v2.functional.crop(img, top=top, left=left, height=min_side, width=min_side)
        return img