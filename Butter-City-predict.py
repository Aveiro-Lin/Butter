from ultralytics import YOLO
import argparse
from ultralytics.utils.torch_utils import de_parallel, get_flops_with_torch_profiler

def main(opt):
    model_path = opt.model_path
    name = opt.name

    yolo = YOLO(model_path,task="detect")

    # info = get_flops_with_torch_profiler(yolo.cuda(), imgsz=640)
    # print(info)
    yolo.info()

    result = yolo(source='/data3/liyang/Proj/Cityscapes/images/val',save=True,save_conf = True,save_txt = True,name=name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="runs/detect/All1-City/weights/best.pt")
    parser.add_argument('--name', type=str, default="City-predict")
    opt = parser.parse_args()

    main(opt)