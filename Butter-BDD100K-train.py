from ultralytics import YOLO
import argparse

def main(opt):
    model_path = opt.model_path
    device = opt.device
    name = opt.name

    device = [int(d) for d in device.split(",")]
    if len(device) == 1: device = device[0]

    # model = YOLO('./runs/detect/train4/weights/best.pt')
    model = (YOLO(model_path))
    # Train the moder
    model.train(data='Butter-BDD100K-data.yaml',workers=8,epochs=300,batch=128,device=device,name=name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="ultralytics/cfg/models/Butter/Butter-All-BDD100K.yaml")
    parser.add_argument('--device', type=str, default="4,5,6,7")
    parser.add_argument('--name', type=str, default="BDD100K-train")
    opt = parser.parse_args()

    main(opt)
