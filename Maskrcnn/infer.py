#张晓明


import os
import sys
import random
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import transforms
import argparse
from PIL import Image
import cv2
import numpy as np

from utils import (
    overlay_ann,
    # overlay_mask,
    show
)

seed = 1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#python infer.py --image_path img/4.png --output_path output/

# CATEGORIES2LABELS = {
#     0: "bg",
#     1: "text",
#     2: "title",
#     3: "list",
#     4: "table",
#     5: "image"
# }
CATEGORIES2LABELS = {
    0: "bg",
    1: "word",
    2: "table",
    3: "picture",
    4: "title",
}
SAVE_PATH = "output/"

#MODEL_PATH = "./model_196000.pth"   #模型参数
MODEL_PATH = "./T/my_model.pth"

#MODEL_PATH = "../model_final.pkl"
'''裁剪指定图片中类型的截图'''
# def crop_img(box, filename):
#
#
#             #获取坐标
#             x_1 = box[0]
#             y_1 = box[1]
#             x_2 = box[2]
#             y_2 = box[3]
#
#             img = Image.open(filename)
#             box = (x_1, y_1, x_2, y_2)
#             im2 = img.crop(box)
#             im2.save('./img/crop.png')
'''
模型训练三部分 
     坐标
     标签
     掩码
'''
def get_instance_segmentation_model(num_classes):
    #mask和数量一致
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)   #获取模型架构
    #print("=========model",model)

    # (cls_score): Linear(in_features=1024, out_features=91, biasbias=True)
    '''  in_features：每个输入（x）样本的特征的大小
    　　out_features：每个输出（y）样本的特征的大小'''

    in_features = model.roi_heads.box_predictor.cls_score.in_features   #获取模型的输入特征数in_features=1024
    print("==========in", model.roi_heads.box_predictor)
    '''==========in FastRCNNPredictor(
  (cls_score): Linear(in_features=1024, out_features=91, bias=True)
  (bbox_pred): Linear(in_features=1024, out_features=364, bias=True)
   )'''
   #修改模型的框架参数
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print("model.roi_heads.box_predictor===",model.roi_heads.box_predictor)
    '''经过FastRCNNPredictor后输出的特征数由91变为了6
    model.roi_heads.box_predictor=== FastRCNNPredictor(
      (cls_score): Linear(in_features=1024, out_features=6, bias=True)
      (bbox_pred): Linear(in_features=1024, out_features=24, bias=True)
      )
      '''

    print("model.roi_heads.mask_predictor==",model.roi_heads.mask_predictor)

    ''' 一开始的模型mask的参数  输出的类别为91个
    model.roi_heads.mask_predictor== MaskRCNNPredictor(
  (conv5_mask): ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))
  (relu): ReLU(inplace=True)
  (mask_fcn_logits): Conv2d(256, 91, kernel_size=(1, 1), stride=(1, 1))
)
'''
    #获取输入特征的，mask
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    print(in_features_mask)
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(    #把模型的参数改为

        in_features_mask,
        hidden_layer,
        num_classes
    )
    '''经过MaskRCNNPredictor 改成后的 输出的类别为六个
    MaskRCNNPredictor(
(conv5_mask): ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))
(relu): ReLU(inplace=True)
(mask_fcn_logits): Conv2d(256, 6, kernel_size=(1, 1), stride=(1, 1))
)'''
    print(model.roi_heads.mask_predictor)
    '''
    =============== MaskRCNN(
   (transform): GeneralizedRCNNTransform(
      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      Resize(min_size=(800,), max_size=1333, mode='bilinear')
    )
 
    '''
    return model


def main():
    num_classes = 5
    model = get_instance_segmentation_model(num_classes)

    #model.cuda()
    model.to(device)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", 
        default = './model/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
        type = str,
        help = "model checkpoint directory"
    )
    parser.add_argument(
        "--image_path",
        #default = None,
        default='./img/4.png',
        type = str,
        required = True,
        help = "directory of path of image to be passed into model"

    )
    parser.add_argument(
        "--output_path",
        default='./img/',
        type  = str,
        required = True,
        help = "output directory of model results"
    )

    args = parser.parse_args()
    print(args)
    '''
    Namespace(image_path='img/4.png', model_path='./model/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth', output_path='output/')
    '''


    #args = parser.parse_known_args()
    #checkpoint_path = args.model_path
    if os.path.exists(MODEL_PATH):

        checkpoint_path = MODEL_PATH
    else:
        checkpoint_path = args.model_path

    print(checkpoint_path)
    assert os.path.exists(checkpoint_path)    #找到模型参数路径  往下执行
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    #print("==============checkpoint",checkpoint)
    #print("==============checkpoint['model']",checkpoint['model'])
    #model.load_state_dict(checkpoint['model'], True)   #权重给到模型

    model.load_state_dict(checkpoint)
    #model.load_state_dict(torch.load(checkpoint)) #表示将训练好的模型参数重新加载至网络模型中


    model.eval()

    # NOTE: custom  image
    if len(argv) > 0 and os.path.exists(args.image_path):
        image_path = args.image_path     #用的这个
    else:
        image_path = filename

    print("================",image_path)
    #assert os.path.exists(image_path)

    image = cv2.imread(image_path)
    rat = 1000 / image.shape[0]

    image = cv2.resize(image, None, fx=rat, fy=rat) #如果fx=1.44，则将原图片的x轴缩小为原来的1.44倍，将y轴缩小为原来的1.44倍
    '''transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起
    transforms. ToTensor()将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，
    其将每一个数值归一化到[0,1]，其归一化方法比较简单，直接除以255即可
    '''
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    image = transform(image)
    #print("========image.to(device)",image.to(device))
    with torch.no_grad():
        prediction = model([image.to(device)])
        print(prediction)
            #预测值

    image = torch.squeeze(image, 0).permute(1, 2, 0).mul(255).numpy().astype(np.uint8)

    for pred in prediction:
        for idx, mask in enumerate(pred['masks']):
            if pred['scores'][idx].item() < 0.7:       #得分小于0.6 不要
                continue

            # m = mask[0].mul(255).byte().cpu().numpy()
            #[95, 55, 674, 933] [12, 71, 688, 337]
            #获取坐标
            box = list(map(int, pred["boxes"][idx].tolist()))
            print(box)
            '''
            [ 165.9319,  623.9207,  780.0000,  948.6614]=>
            CATEGORIES2LABELS = {
            0: "bg",
            1: "text",
            2: "title",
            3: "list",
            4: "table",
            5: "image"
            }'''
            label = CATEGORIES2LABELS[pred["labels"][idx].item()]
            print(label)
            # crop_img(box, filename)
            score = pred["scores"][idx].item()


            image = overlay_ann(image, box, label, score)
            print(idx)
    # cv2.imwrite('/home/z/research/publaynet/example_images/{}'.format(os.path.basename(image_path)), image)
    # if os.path.exists(args.output_path):
    #     cv2.imwrite(args.output_path+'/{}'.format(os.path.basename(image_path)), image)
    # else:
    #     os.mkdir(args.output_path)
    #     cv2.imwrite(args.output_path+'/{}'.format(os.path.basename(image_path)), image)

    show(image)


if __name__ == "__main__":
    import sys

    #裁剪的照片
    filename='./img/crop.png'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    argv = sys.argv[1:]
    print("==============argv",argv)
    #main(argv)
    main()
