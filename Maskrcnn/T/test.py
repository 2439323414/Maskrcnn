# 张晓明
import train
# pick one image from the test set
from PIL import Image
import torch
Image.open('./PennFudanPed/PNGImages/0000001.png')

mask = Image.open('PennFudanPed/PedMasks/0000001_mask.png')

mask.putpalette([
    0, 0, 0,  # black background
    255, 255, 0,  # index 1 is red
    255, 255, 0,  # index 2 is yellow
    255, 153, 0,  # index 3 is orange
])
masks = torch.as_tensor(mask, dtype=torch.uint8)
print(masks)
mask.show()