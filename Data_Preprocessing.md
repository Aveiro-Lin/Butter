# <div align="center">Data Preprocessing</div>


> The Butter model adopts the data preprocessing approach used in the YOLO series of models.






### `KITTI` Dataset

```bash
cd ./Butter/datasets/KITTI
python split.py
python kitti2yolo.py
```



### `Cityscapes` Dataset

```bash
cd ./Butter/datasets/cityscapes
python cityscapes2yolo.py
```



### `BDD100K` Dataset

```bash
cd ./Butter/datasets/bdd100k
python bdd100k2voc.py
python voc2yolo.py
```




### `COCO` Dataset

```bash
cd ./Butter/datasets/coco
python coco2yolo.py --json_path labels/annotations/instances_train2017.json --save_path labels/train2017
python coco2yolo.py --json_path labels/annotations/instances_val2017.json
 --save_path labels/val2017
```





