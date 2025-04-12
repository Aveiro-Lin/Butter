from ultralytics import YOLO
import argparse

def main(opt):
    model_path = opt.model_path
    name = opt.name

    yolo = YOLO(model_path,task="detect")

    yolo.info()

    result = yolo(source='datasets/coco/images/test2017',save=True,save_conf = True,save_txt = True,name=name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="runs/detect/coco-all-1182/weights/best.pt")
    parser.add_argument('--name', type=str, default="COCO-predict")
    opt = parser.parse_args()

    main(opt)
