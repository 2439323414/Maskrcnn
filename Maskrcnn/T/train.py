# 张晓明

import coco_utils
import coco_eval

from torch.utils import data

from api import *
import utils
import torchvision
import transforms as T
from engine import train_one_epoch, evaluate
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

#图像增强
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)


def get_instance_segmentation_model(num_classes):
    # mask和数量一致
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)  # 获取模型架构

    # (cls_score): Linear(in_features=1024, out_features=91, biasbias=True)
    '''  in_features：每个输入（x）样本的特征的大小
    　　out_features：每个输出（y）样本的特征的大小'''

    in_features = model.roi_heads.box_predictor.cls_score.in_features  # 获取模型的输入特征数in_features=1024

    # 修改模型的框架参数
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    # 获取输入特征的，mask
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    #print(in_features_mask)
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(  # 把模型的参数改为

        in_features_mask,
        hidden_layer,
        num_classes
    )
    print(model.roi_heads.mask_predictor)

    return model

#模型训练
def main():
    num_classes=5

    # use the PennFudan dataset and defined transformations
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=False))

    #dataset = PennFudanDataset('PennFudanPed')
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))
    #dataset_test = PennFudanDataset('PennFudanPed')
    # split the dataset in train and test set  在训练集和测试集中拆分数据集
    torch.manual_seed(1)   #随机
    indices = torch.randperm(len(dataset)).tolist()


    dataset = torch.utils.data.Subset(dataset, indices[:-20])
    #print("========",len(indices[:-20]),indices[:-20])

    dataset_test = torch.utils.data.Subset(dataset_test, indices[-20:])
    #print("========", len(indices[-20:]), indices[-20:])
    # define training and validation data loaders定义训练和验证数据加载器
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # the dataset has two classes only - background and person
    #num_classes = 2

    # get the model using the helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer  构造优化器
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005,
    #                             momentum=0.9, weight_decay=0.0005)
    #采用adam优化器进行优化
    optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999))
    print(optimizer)
    # the learning rate scheduler decreases the learning rate by 10x every 3 epochs 学习速率计划程序每 3 个 epoch 将学习速率降低 10 倍
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.1)

    # training
    num_epochs = 10
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        #loss rpn网络的两个损失，分类的两个损失，以及mask分支的损失函数
        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset  在测试数据集上评估
        evaluate(model, data_loader_test, device=device)
    print("hello zxm")



    torch.save(model.state_dict(), "my_model.pth")

    # model.load_state_dict(torch.load("my_model.pth"))




if __name__ == '__main__':
    main()

