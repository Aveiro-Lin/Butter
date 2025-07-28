import json
import os
 
from sympy import print_glsl
import shutil
 
# Class list and class dictionary
all_classes = ['car', 'person', 'rider', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
class_dict = {'car': 0, 'person': 1, 'rider': 2, 'truck': 3, 'bus': 4, 'train': 5, 'motorcycle': 6, 'bicycle': 7}

# Root directory
# rootdir = 'D:/DeepLearningDatasets/ObjectDetection/Cityscapes/gtFine_trainvaltest/gtFine/test'
rootdir = '/home/xiaojianlin/Project/Butter/datasets/cityscapes/labels/gtFine/train'
# rootdir = '/home/xiaojianlin/Project/Butter/datasets/cityscapes/labels/gtFine/val'

# Output directory
output_rootdir = '/home/xiaojianlin/Project/Butter/datasets/cityscapes/labels/leftImg8bit/train'
# output_rootdir = '/home/xiaojianlin/Project/Butter/datasets/cityscapes/labels/leftImg8bit/val'
if not os.path.exists(output_rootdir):
    os.makedirs(output_rootdir, exist_ok=True)
else:
    shutil.rmtree(output_rootdir)
 
 
def position(pos):
    x = [point[0] for point in pos]
    y = [point[1] for point in pos]
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)
    return float(x_min), float(x_max), float(y_min), float(y_max)
 
 
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh
 
 
def convert_annotation(json_id, city_name):
    json_file_path = os.path.join(rootdir, city_name, '%s.json' % json_id)
    out_file_path = os.path.join(output_rootdir, city_name, '%s.txt' % json_id)
    out_file_path = out_file_path.replace('_gtFine_polygons', '_leftImg8bit')

    if not os.path.exists(os.path.dirname(out_file_path)):
        os.makedirs(os.path.dirname(out_file_path))
 
    with open(json_file_path, 'r') as load_f:
        load_dict = json.load(load_f)
 
    w = load_dict['imgWidth']
    h = load_dict['imgHeight']
    objects = load_dict['objects']
 
    with open(out_file_path, 'w') as out_file:
        for obj in objects:
            labels = obj['label']
            if labels in class_dict:
                pos = obj['polygon']
                b = position(pos)
                bb = convert((w, h), b)
                cls_id = class_dict[labels]
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
 
 
def jsons_id(rootdir):
    a = []
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            if filename.endswith('.json'):
                filename_without_ext = os.path.splitext(filename)[0]
                a.append(filename_without_ext)
    return a
 
 


# Get all subdirectories
subdirs = [d for d in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, d))]
# print(subdirs)
# ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf', 'erfurt', 'hamburg', 'hanover', 'jena',
# 'krefeld', 'monchengladbach', 'strasbourg', 'stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']

# Generate YOLO-format annotation files for each subdirectory


for subdir in subdirs:
    names = jsons_id(os.path.join(rootdir, subdir))
    for json_id in names:
        convert_annotation(json_id, subdir)