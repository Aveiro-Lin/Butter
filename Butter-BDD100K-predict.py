from ultralytics import YOLO
import argparse

def main(opt):
    model_path = opt.model_path
    name = opt.name

    yolo = YOLO(model_path,task="detect")

    result = yolo(source='datasets/BDD100K/images/test',save=True,save_conf = True,save_txt = True,name=name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="runs/detect/2-BDD-ALL/weights/best.pt")
    parser.add_argument('--name', type=str, default="BDD100K-predict")
    opt = parser.parse_args()

    main(opt)
