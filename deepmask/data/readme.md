# COCO

### Download raw images and annotations

````shell
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip ./train2017.zip
unzip ./val2017.zip
unzip ./annotations_trainval2017.zip
cd ../loader/pycocotools && make && cd ..
````
