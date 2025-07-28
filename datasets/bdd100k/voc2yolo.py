import glob
import xml.etree.ElementTree as ET
import os
import shutil
 
#种类有'car', 'bus', 'person', 'bike', 'truck', 'motor', 'train', 'rider', 'traffic sign', 'traffic light'，可以自己定义序号
# class_names = ['car', 'bus', 'person', 'bike', 'truck', 'motor', 'train', 'rider', 'traffic sign', 'traffic light']
class_names = ["bus", "traffic light", "traffic sign", "person", "bike", "truck", "motor", "car", "train", "rider"]
 
 
def single_xml_to_txt(xml_file, dstDir):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    txt_file = dstDir + os.path.basename(xml_file).split('.')[0] + ".txt"
    with open(txt_file, 'w') as txt_file:
        for member in root.findall('object'):
            picture_width = int(root.find('size')[0].text)
            picture_height = int(root.find('size')[1].text)
            class_name = member[0].text
            class_num = class_names.index(class_name)
            box_x_min = int(member[4][0].text)  
            box_y_min = int(member[4][1].text)  
            box_x_max = int(member[4][2].text)  
            box_y_max = int(member[4][3].text)  
            x_center = (box_x_min + box_x_max) / (2 * picture_width)
            y_center = (box_y_min + box_y_max) / (2 * picture_height)
            width = (box_x_max - box_x_min) / (2 * picture_width)
            height = (box_y_max - box_y_min) / (2 * picture_height)
            print(class_num, x_center, y_center, width, height)
            txt_file.write(str(class_num) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(
                height) + '\n')
 
 
def dir_xml_to_txt(path, dstDir):
    if not os.path.exists(dstDir):
        os.makedirs(dstDir, exist_ok=True)
    else:
        shutil.rmtree(dstDir)
    i = 1
    for xml_file in glob.glob(path + '*.xml'):
        single_xml_to_txt(xml_file, dstDir)
        i += 1
 
 
def main(path, dstDir):
    dir_xml_to_txt(path, dstDir)
 
if __name__ == '__main__':
    srcDir = './labels/100k/val_xml/'
    dstDir = './labels/100k/val_txt/'
    main(srcDir, dstDir)
    
    srcDir = './labels/100k/train_xml/'
    dstDir = './labels/100k/train_txt/'
    main(srcDir, dstDir)