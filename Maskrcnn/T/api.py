# 张晓明
import os
import torch
import numpy as np
import torch.utils.data
from PIL import Image

import yaml

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned加载所有图像文件，对它们进行排序以确保它们对齐
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        self.yaml = list(sorted(os.listdir(os.path.join(root, "yaml"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])

        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        yaml_path = os.path.join(self.root, "yaml", self.yaml[idx])
        print(idx,img_path,mask_path,yaml_path)
        file = open(yaml_path, 'r', encoding="utf-8")
        temp = file.read()
        with open(yaml_path) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]

        for i in range(len(labels)):
            if labels[i] == 'word':
                labels[i] = 1
            if labels[i] == 'table':
                labels[i] = 2
            if labels[i] == 'picture':
                labels[i] = 3
            if labels[i] == 'title':
                labels[i] = 4
       # print(labels)

        img = Image.open(img_path).convert("RGB")

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        #obj_ids = np.unique(mask)
        #删除重复元素并排序，返回排序后的数组和索引值
        obj_ids,s = np.unique(mask, return_index=True)
        # print(obj_ids)
        # print(s)
        #求排序后对应的索引值
        obj_idss = np.argsort(s)
        #print(obj_idss)
        # first id is the background, so remove it
        obj_ids = obj_idss[1:]
        # split the color-encoded mask into a set of binary masks[1,2,3,4]
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask 个数
        num_objs = len(obj_ids)

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])

            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        # print(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 标签
        labels = torch.as_tensor(labels,dtype=torch.int64)
        #labels = torch.ones((num_objs,), dtype=torch.int64)
        #print(labels)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        #print(masks)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)


        return img, target

    def __len__(self):
        return len(self.imgs)



# dataset = PennFudanDataset('PennFudanPed/')
#
# print(dataset)