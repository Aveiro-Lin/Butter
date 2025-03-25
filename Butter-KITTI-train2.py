from ultralytics import YOLO
# import os
import torch
import argparse

def main(opt):
    model_path = opt.model_path
    device = opt.device
    name = opt.name

    device = [int(d) for d in device.split(",")]
    if len(device) == 1: device = device[0]

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # device = torch.device('cuda：0' if torch.cuda.is_available() and use_cuda else 'cpu')

    # Load a model
    # model = YOLO('./runs/detect/train4/weights/best.pt')
    model = (YOLO(model_path))

    # Train the moder
    model.train(data='Butter-KITTI-data.yaml',workers=0,epochs=300,batch=32,device=device,name=name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="ultralytics/cfg/models/Butter/Butter-FAFCE.yaml")
    parser.add_argument('--device', type=str, default=0)
    parser.add_argument('--name', type=str, default="KITTI-train")
    opt = parser.parse_args()

    main(opt)
