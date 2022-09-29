import argparse
import torch
from models import *
import torchvision
from PIL.Image import Image
import os
import tqdm
from utils.metric import *
from utils.helpers import *


def predict_dataloader(model, test_loader, num_classes, device, criterion=None, precision='fp32', print_detail=True,
                 auc_roc=False, ignore_labels=None, save=None, show_img=False):
    model.eval()
    intersect = torch.zeros(num_classes).to(device)
    pred_area = torch.zeros(num_classes).to(device)
    label_area = torch.zeros(num_classes).to(device)
    tbar = tqdm(test_loader, ncols=130, position=0, leave=True)
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(tbar):
            if precision == 'fp16':
                with torch.cuda.amp.autocast(enabled=True):
                    if not model.depth:
                        preds = model(input[0])
                    else:
                        preds = model(*input)
                    if criterion is not None:
                        loss = criterion(preds, target)
            else:
                if not model.depth:
                    preds = model(input[0])
                else:
                    preds = model(*input)
                if criterion is not None:
                    preds = upscale(preds)
            if isinstance(preds, tuple):
                preds = preds[0]
            pred = torch.argmax(preds, dim=1, keepdim=True).squeeze()
            inter, pPred, pTarget = calculate_hist(pred, target, preds.shape[1], ignore_labels=ignore_labels)
            intersect = torch.add(inter, intersect)
            pred_area = torch.add(pPred, pred_area)
            label_area = torch.add(pTarget, label_area)




    class_iou, miou = meanIoU(aInter=intersect, aPreds=pred_area, aLabels=label_area)
    acc, class_precision, class_recall = class_measurement(aInter=intersect, aPreds=pred_area, aLabels=label_area)
    log = {
        'miou': miou,
        'class_iou': class_iou.cpu().numpy(),
        'class_precision': class_precision.cpu().numpy(),
    }
    return log



def predict(model, img, img_depth=None, save=""):
    filename, file_extension = os.path.splitext('/path/to/somefile.ext')
    if file_extension in ["jpg", "png"]:
        video = False
    else
    if




if __name__ == "__main__":
    model = CustomModel2(num_classes=19)
    model.load_state_dict(torch.load("C:\\Users\\Florian\\PycharmProjects\\SegmentationProjet\\saved\\08_24-0932_32\\best_model.pth"))
    # Initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    img = Image.load("E:\\Dataset\\leftImg8bit_trainvaltest\\leftImg8bit\\test\\berlin\\berlin_000001_000019_leftImg8bit.png")
    depth = Image.load("E:\\Dataset\\leftImg8bit_trainvaltest\\disparity\\test\\berlin\\berlin_000001_000019_leftImg8bit.png")

    for img, depth in dataset:
        img, depth = transform(img, depth)
        pred = model(img, depth)
        pred = torchvision.utils.draw_segmentation_masks


class CSPOSA_Module(nn.Module):
    def __init__(self, in_channels, stage_ch, out_channels, first_stage_ch=None, identity=False, depthwise=False):
        super(CSPOSA_Module, self).__init__()
        self.identity = identity
        self.layers = []
        # feature aggregation
        self.first_stage_ch = first_stage_ch
        in_channel = in_channels
        for i in range(2):
            self.layers.append(ConvBNRelu(in_channels=in_channel, out_channels=stage_ch, kernel_size=3,
                                          groups=1, padding=1, bias=False))
            in_channel = stage_ch
        in_channel = in_channels + 2 * stage_ch
        if self.first_stage_ch is not None:
            in_channel = in_channel + self.first_stage_ch
        self.concat = ConvBNRelu(in_channels=in_channel, out_channels=out_channels, kernel_size=1, padding=0, stride=0,
                                 bias=False)

    def forward(self, x, x_first=None):
        output = []
        if self.first_stage_ch is not None:
            output.append(x_first)
        output.append(x)
        identity_feat = x
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        xt = self.concat(x)
        if self.identity:
            xt = xt + identity_feat
        return xt


class CSPOStage(nn.Module):
    def __init__(self, in_channels, stage_ch, concat_ch):
        super(CSPOStage, self).__init__()

    def forward(self, x):
