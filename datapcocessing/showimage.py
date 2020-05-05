# _*_ coding: utf-8 _*_


import json
import pandas as pd
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, sys, glob
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
# from data_aug.data_aug import *
from PIL import Image
import math
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch import autograd
# %matplotlib inline

# In[33]:


# 定义全局变量
TRAIN_JSON_PATH = './complex_trash.json'
TRAIN_IMAGE_PATH = 'D:\\biendata\\' # 训练图片的路径
# VAL_IMAGE_PATH = "work/val/images" # 验证图片的路径
# CKP_PATH = 'data/data22244/fasterrcnn.pth' # 预训练的权重
# SAVE_PATH = "result.json" # 提交结果的保存路径
# MODEL_PATH = "model.pth" # 模型的保存路径
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(TRAIN_JSON_PATH,'r',encoding='utf-8') as f:
    train_json = json.load(f)

NUM_CLASS = len(train_json["categories"]) + 1
print('NUM_CLASS=',NUM_CLASS)
id_to_label = dict([(item["id"], item["name"]) for item in train_json["categories"]]) # 将ID转成中文名称
print(id_to_label)
image_paths = glob.glob(os.path.join(TRAIN_IMAGE_PATH, "*.png"))[-2:-1] # 为了演示，只取前100张
imageid_of_annotation = list(set([i["image_id"] for i in train_json["annotations"]]))
print(imageid_of_annotation)
image_paths = [
    i for i in image_paths if os.path.basename(i).split(".")[0] in imageid_of_annotation
] # 剔除image_id不在annotation里面的图片
# print(image_paths)
# len_train = int(0.8 * len(image_paths))
# train_image_paths = image_paths[:len_train]
# eval_image_paths = image_paths[len_train:]
# val_image_paths = glob.glob(os.path.join(VAL_IMAGE_PATH, "*.png"))[:10]# 为了演示，只取前10张

# In[5]:


# 将所有的标注信息整合到DataFrame里，方便后续分析
data = []
for idx, annotations in enumerate(train_json['annotations']):
    data.append([annotations['image_id'],annotations['category_id'], idx])
print(data)
data = pd.DataFrame(data, columns=['image_id','category_id', 'idx'])
data['category_name'] = data['category_id'].map(id_to_label)
print('训练集总共有多少张图片？ ', len(train_json['images']))
print('训练集总共有多少个标注框？ ',len(train_json['annotations']))
print('总共有多少种类别？ ',len(train_json['categories']))
print(data.head())

# #### 分组显示图上bounding box数量的图片数量  
# 每张图片上最少有1个bounding box, 最多有20个  
# 拥有5个bounding box的图片数量最多，平均每张图片上有9个bounding box  

# In[6]:


image_group = data.groupby('image_id').agg({'idx':'count'})['idx']
# print('image_group=',image_group)
image_group.value_counts().sort_index().plot.bar()  #
print(image_group.describe())
plt.xlabel("numbers of bboxs per image")
plt.ylabel('numbers of images')
plt.show()
# #### 显示每个category出现的次数
# 每个category最少出现1次，最多出现897次，平均出现196次  
# 出现次数最多的category分别为金属工具、塑料包装、药瓶等

# In[7]:


category_group = data.groupby('category_id').agg({'idx':'count'})['idx']# .value_counts()
category_group.sort_index().plot.bar()
print(category_group.describe())
s = category_group.sort_values(ascending=True).reset_index().head(54)
plt.show()
# print(s)
# print(s['category_id'].values.tolist())


# In[8]:


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
# 显示图片及bounding box的辅助函数
def compute_color_for_labels(label): #不同label显示不同颜色
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]

    return tuple(color)

def draw_boxes(img, bbox, identities, offset=(0,0)):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        print(label)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        print(t_size)
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img
# 将coco的bounding box格式从[x,y,width,height]转成[x1,y1,x2,y2]
def xywh_to_xyxy(points):   #得出对角线坐标
    return [points[0], points[1], points[0] + points[2], points[1]+ points[3]]

# #### 图像增强的方法
# 为了最大化的利用数据，对训练数据进行增强是一个有效手段  
# 以下代码显示原始的图片，以及进行旋转、平移后的图片  
# 图像增强的关键在于图片进行变换后，bounding box的坐标要跟着变换  
# 除了介绍的这两种，还有其他不同的图像增强方法  

# In[11]:



show_image_path = image_paths[0]
show_image_id = os.path.basename(show_image_path).split('.')[0]
show_annotation = [i for i in train_json['annotations'] if i['image_id'] == show_image_id]

def show_original_image(show_image_path, show_annotation):


    show_bboxes = [xywh_to_xyxy(a['bbox']) for a in show_annotation]
    show_bboxes = np.array(show_bboxes).astype(np.float32)
    print('show_bboxs',show_bboxes)
    show_labels = [a['category_id'] for a in show_annotation]
    print(show_labels)
    show_iamge = cv2.imread(show_image_path) # [:,:,::-1]
    show_iamge = cv2.cvtColor(show_iamge, cv2.COLOR_BGR2RGB)
    show_iamge = draw_boxes(show_iamge, show_bboxes, show_labels)
    print('显示原始的图片')
    plt.imshow(show_iamge)
    plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
show_original_image(show_image_path, show_annotation)

