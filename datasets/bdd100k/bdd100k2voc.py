import os
import json
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
from xml.dom.minidom import parseString
import shutil
 
#种类有'car', 'bus', 'person', 'bike', 'truck', 'motor', 'train', 'rider', 'traffic sign', 'traffic light'，可以自己定义序号
# categorys = ['car', 'bus', 'person', 'bike', 'truck', 'motor', 'train', 'rider', 'traffic sign', 'traffic light']
categorys = ["bus", "traffic light", "traffic sign", "person", "bike", "truck", "motor", "car", "train", "rider"]
 
def parseJson(jsonFile):
    '''
      params:
        jsonFile -- BDD00K数据集的一个json标签文件
      return:
        返回一个列表的列表，存储了一个json文件里面的方框坐标及其所属的类，
        形如：[[325, 342, 376, 384, 'car'], [245, 333, 336, 389, 'car']]
    '''
    objs = []
    obj = []
    f = open(jsonFile)
    info = json.load(f)
    objects = info['frames'][0]['objects']
    for i in objects:
        if (i['category'] in categorys):
            obj.append(int(i['box2d']['x1']))
            obj.append(int(i['box2d']['y1']))
            obj.append(int(i['box2d']['x2']))
            obj.append(int(i['box2d']['y2']))
            obj.append(i['category'])
            objs.append(obj)
            obj = []
    
    return objs
 
 
class PascalVocWriter:
 
    def __init__(self, foldername, filename, imgSize, databaseSrc='Unknown', localImgPath=None):
        '''
        params:
          foldername -- 要存储的xml文件的父目录
          filename -- xml文件的文件名
          imgSize -- 图片的尺寸
          databaseSrc -- 数据库名，这里不需要，默认为Unknown
          localImaPath -- xml文件里面的<path></path>标签的内容
      '''
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
 
    def prettify(self, elem):
        """
        params:
          elem -- xml的根标签，以<annotation>开始
        return:
          返回一个美观输出的xml（用到minidom），本质是一个str
        """
        xml = ElementTree.tostring(elem)
        dom = parseString(xml)
        
        prettifyResult = dom.toprettyxml('    ')
        return prettifyResult
 
    def genXML(self):
        """
        return:
          生成一个VOC格式的xml，返回一个xml的根标签，以<annotation>开始
        """
        
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None or \
                len(self.boxlist) <= 0:
            return None
 
        top = Element('annotation')  
        folder = SubElement(top, 'folder')
        folder.text = self.foldername  
 
        filename = SubElement(top, 'filename')  
        filename.text = self.filename  
 
        localImgPath = SubElement(top, 'path')  
        localImgPath.text = self.localImgPath  
 
        source = SubElement(top, 'source')  
        database = SubElement(source, 'database')  
        database.text = self.databaseSrc  
 
        size_part = SubElement(top, 'size')  
        width = SubElement(size_part, 'width')  
        height = SubElement(size_part, 'height')  
        depth = SubElement(size_part, 'depth')  
        width.text = str(self.imgSize[1])  
        height.text = str(self.imgSize[0])  
        if len(self.imgSize) == 3:  
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'
 
        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top
 
    def addBndBox(self, xmin, ymin, xmax, ymax, name):
        '''
        将检测对象框坐标及其对象类别作为一个字典加入到self.boxlist中
        params:
          xmin -- 检测框的左上角的x坐标
          ymin -- 检测框的左上角的y坐标
          xmax -- 检测框的右下角的x坐标
          ymax -- 检测框的右下角的y坐标
          name -- 检测框内的对象类别名
        '''
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        self.boxlist.append(bndbox)
 
    def appendObjects(self, top):
        '''
        在xml文件中加入检测框的坐标及其对象类别名
        params:
          top -- xml的根标签，以<annotation>开始
        '''
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            name.text = str(each_object['name'])
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            truncated.text = "0"
            difficult = SubElement(object_item, 'Difficult')
            difficult.text = "0"
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])
 
    def save(self, targetFile=None):
        '''
        以美观输出的xml格式来保存xml文件
        params:
          targetFile -- 存储的xml文件名，不包括.xml部分
        '''
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        subdir = self.foldername.split('/')[-1]
        if not os.path.isdir(subdir):
            os.mkdir(subdir)
        if targetFile is None:
            with open(self.foldername + '/' + self.filename + '.xml', 'w') as out_file:
                prettifyResult = self.prettify(root)
                out_file.write(prettifyResult)
                out_file.close()
        else:
            with open(targetFile, 'w') as out_file:
                prettifyResult = self.prettify(root)
                out_file.write(prettifyResult)
                out_file.close()
 
class PascalVocReader:
    def __init__(self, filepath):
        self.shapes = []
        self.filepath = filepath
        self.parseXML()
    def getShapes(self):
        return self.shapes
    def addShape(self, label, bndbox):
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        self.shapes.append((label, points, None, None))
    def parseXML(self):
        assert self.filepath.endswith('.xml'), "Unsupport file format"
        parser = etree.XMLParser(encoding='utf-8')
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        filename = xmltree.find('filename').text
        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find("bndbox")
            label = object_iter.find('name').text
            self.addShape(label, bndbox)
        return True
 
 
def main(srcDir, dstDir):
    if not os.path.exists(dstDir):
        os.makedirs(dstDir, exist_ok=True)
    else:
        shutil.rmtree(dstDir)
    i = 1
    for dirpath, dirnames, filenames in os.walk(srcDir):
        for filepath in filenames:
            fileName = os.path.join(dirpath, filepath)
            print(fileName)
            print("processing: {}, {}".format(i, fileName))
            i = i + 1
            xmlFileName = filepath[:-5]  
            
            objs = parseJson(str(fileName))
            
            if len(objs):
                tmp = PascalVocWriter(dstDir, xmlFileName, (720, 1280, 3), fileName)
                for obj in objs:
                    tmp.addBndBox(obj[0], obj[1], obj[2], obj[3], obj[4])
                tmp.save()
            else:
                print(fileName)
 
 
if __name__ == '__main__':
    srcDir = './labels/100k/val'  
    dstDir = './labels/100k/val_xml'

    main(srcDir, dstDir)

    srcDir = './labels/100k/train'  
    dstDir = './labels/100k/train_xml'

    main(srcDir, dstDir)
