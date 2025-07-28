# -*- coding: utf-8 -*-


"""
2021/1/24
Convert a COCO-format dataset to YOLO-format. The original code uses a traversal approach, which is too slow.
Here, the time complexity is improved from O(nm) to O(n+m), at the cost of increased memory usage.
--json_path: Path to the input JSON file
--save_path: Name of the folder to save the output files, default is 'labels' in the current directory.
"""


import os 
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', default='./instances_val2017.json',type=str, help="input: coco format(json)")
parser.add_argument('--save_path', default='./labels', type=str, help="specify where to save the output dir of labels")
arg = parser.parse_args()

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

if __name__ == '__main__':

    json_file = arg.json_path  # COCO Object Instance annotation file
    ana_txt_save_path = arg.save_path  # Path to save the output

    data = json.load(open(json_file, 'r'))
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)

    id_map = {}  # COCO dataset category IDs are not continuous! Remap them before output!
    for i, category in enumerate(data['categories']): 
        id_map[category['id']] = i

    # Build lookup tables in advance to reduce time complexity

    max_id = 0
    for img in data['images']:
        max_id = max(max_id, img['id'])

    img_ann_dict = [[] for i in range(max_id+1)] 
    for i, ann in enumerate(data['annotations']):
        img_ann_dict[ann['image_id']].append(i)

    for img in tqdm(data['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # Corresponding txt file name, same as jpg        

        f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
        '''for ann in data['annotations']:
            if ann['image_id'] == img_id:
                box = convert((img_width, img_height), ann["bbox"])
                f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))'''
        # Direct lookup can be used here without repeated traversal
        for ann_id in img_ann_dict[img_id]:
            ann = data['annotations'][ann_id]
            box = convert((img_width, img_height), ann["bbox"])
            f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
        f_txt.close()
        
