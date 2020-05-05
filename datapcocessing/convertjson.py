import os
import json
import re
import pandas as pd
import glob
import shutil
import torch
from xlwt import *


def Addsimpimgs(simjsonpath,simimgpath,comjsonpath,comimgpath,newjsonpath):
    with open(comjsonpath,'r',encoding='utf-8') as f:
        train_json = json.load(f)
        id_to_label = dict([(item["id"], item["name"]) for item in train_json["categories"]])
        imageid_of_annotation = list(set([i["image_id"] for i in train_json["annotations"]]))
        data = []
        for idx, annotations in enumerate(train_json['annotations']):
            data.append([annotations['image_id'],annotations['category_id'], idx])
        data = pd.DataFrame(data, columns=['image_id','category_id', 'idx'])
        data['category_name'] = data['category_id'].map(id_to_label)
        category_group = data.groupby('category_id').agg({'idx':'count'})['idx']
        s = category_group.sort_values(ascending=True).reset_index().head(60)
        result = s['category_id'].values.tolist()        #得到图片数量最小的前60个类别的id列表
    f.close()
    with open(simjsonpath,'r',encoding='utf-8') as ff:
        train_json_data = json.load(ff)
        data1 = []
        for idx, annotations in enumerate(train_json_data['annotations']):
            data1.append([annotations['image_id'],annotations['category_id']])
        img_to_mv = [i[0] for i in data1 if i[1] in result]
        for ele in img_to_mv:
            name = str(ele) + '.png'
            shutil.copy(simimgpath+name,comimgpath)
        imgid_to_add = list(j for j in train_json_data['images'] if j['id'] in img_to_mv )
        ann_to_add = list(k for k in train_json_data['annotations'] if k['image_id'] in img_to_mv)   
    ff.close()   
    with open(comjsonpath,'r',encoding='utf-8') as fff:
        train_com_json = json.load(fff)
        temp_dict = {}
        data2 = train_com_json['images']
        data3 = train_com_json['annotations']
        for ele in imgid_to_add:
            data2.append(ele)
        for cont in ann_to_add:
            data3.append(cont)
        temp_dict['licenses'] = train_com_json['licenses']
        temp_dict['info'] = train_com_json['info']
        temp_dict['type'] = train_com_json['type']
        temp_dict['categories'] = train_com_json['categories']
        temp_dict['annotations'] = data3
        temp_dict['images'] = data2    
        with open(newjsonpath,'w',encoding='utf-8') as w:
            json.dump(temp_dict,w)
        w.close()
    fff.close()


def Cocopretrained():
    import torch
    num_classes = 21
    model_coco = torch.load("/cache/common/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth")
        
    # weight
    model_coco["state_dict"]["bbox_head.0.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.0.fc_cls.weight"][:num_classes, :]
    model_coco["state_dict"]["bbox_head.1.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.1.fc_cls.weight"][:num_classes, :]
    model_coco["state_dict"]["bbox_head.2.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.2.fc_cls.weight"][:num_classes, :]
    # bias
    model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][:num_classes]
    model_coco["state_dict"]["bbox_head.1.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.1.fc_cls.bias"][:num_classes]
    model_coco["state_dict"]["bbox_head.2.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.2.fc_cls.bias"][:num_classes]
    #save new model
    torch.save(model_coco,"/cache/common/coco_pretrained_weights_classes_%d.pth"%num_classes)



def coverttrainjson():

    # os.mkdir('/cache/train/train1.json')

    temp_dict = {}
    result1 = []
    result2 = []


    f = open('/cache/train/train_complex.json','r',encoding='UTF-8')
# w = open('./traintest.json','w')
    data = json.load(f)
    # print(data)
    temp_dict['licenses'] = data['lincenses']
    temp_dict['info'] = data['info']
    temp_dict['type'] = data['type']
    temp_dict['categories'] = data['categories']
    l = len(data['images'])
    data1 = data['images']
    for i in range(l):
        img = {}
        img['id'] = data1[i]['image_id']
        img['width'] = data1[i]['width']
        img['height'] = data1[i]['height']
        string = data1[i]['file_name']
        string1 = re.sub(r'images_withoutrect/|png','',string,1)
        img['file_name'] = string1
        result1.append(img)
    temp_dict['images'] = result1    
# print(result1)

    ll = len(data['annotations'])
    data2 = data['annotations']

    for j in range(ll):
        ann = {}
        ann['bbox'] = data2[j]['bbox']
        ann['id'] = j + 1
        ann['area'] = data2[j]['area']
        ann['iscrowd'] = data2[j]['iscrowd']
        ann['category_id'] = data2[j]['category_id']
        ann['image_id'] = data2[j]['image_id']
        result2.append(ann)
# print(result2)

    temp_dict['annotations'] = result2
    with open ('/cache/train/traincom.json','w',encoding='UTF-8') as w:
        json.dump(temp_dict,w)
    w.close()
    f.close()



def covertvaljson():

    # os.mkdir('/cache/train/train1.json')

    temp_dict = {}
    result1 = []
    result2 = []


    f = open('/cache/val/val_open_complex.json','r',encoding='UTF-8')
# w = open('./traintest.json','w')
    data = json.load(f)
    # print(data)
    temp_dict['licenses'] = data['lincenses']
    temp_dict['info'] = data['info']
    temp_dict['type'] = data['type']
    temp_dict['categories'] = data['categories']
    l = len(data['images'])
    data1 = data['images']
    for i in range(l):
        img = {}
        img['id'] = data1[i]['image_id']
        img['width'] = data1[i]['width']
        img['height'] = data1[i]['height']
        string = data1[i]['file_name']
        string1 = re.sub(r'images_withoutrect/|png','',string,1)
        img['file_name'] = string1
        result1.append(img)
    temp_dict['images'] = result1    
# print(result1)

#     ll = len(data['annotations'])
#     data2 = data['annotations']

#     for j in range(ll):
#         ann = {}
#         ann['bbox'] = data2[j]['bbox']
#         ann['id'] = j + 1
#         ann['area'] = data2[j]['area']
#         ann['iscrowd'] = data2[j]['iscrowd']
#         ann['category_id'] = data2[j]['category_id']
#         ann['image_id'] = data2[j]['image_id']
#         result2.append(ann)
# # print(result2)

#     temp_dict['annotations'] = result2
    with open ('/cache/val/valcom.json','w',encoding='UTF-8') as w:
        json.dump(temp_dict,w)
    w.close()
    f.close()

def Addsimpledata(inputfile1,inputfile2,simpicpath,comdatapath,outfilepath):
    with open(inputfile1,'r',encoding='utf-8') as f:
        data = json.load(f)
        simanndata = data['annotations']
        catlist = []
        simdata = []
        for idx , name in enumerate(simanndata):
            simdata.append([name['image_id'],name['category_id']])
        for i in range(len(simanndata)):
            catlist.append(simanndata[i]["category_id"])
        b = set(catlist)
        count = {}
        for i in b:
            count[i] = catlist.count(i)
            # print("单类数据类别{}的图片数为：{}".format(i,catlist.count(i)))
        count[96] = count[121] = count[147] = count[174] = count[175] = 0
    with open(inputfile2,'r',encoding='utf-8') as ff:
        data1 = json.load(ff)
        data2 = data1['annotations']
        imgdata = data1['images']
        catdata = data1['categories']
        licdata = data1['licenses']
        infodata = data1['info']
        typedata = data1['type']
        _list = []
        for i in range(len(data2)):
            _list.append(data2[i]['category_id'])
        count1 = {}
        for j in set(_list):
            count1[j] = _list.count(j)
            print("多类数据类别{}的图片数为：{}".format(j,_list.count(j)))
    countkey = list(count.keys())
    countkey.sort()
    for ele in countkey:
        if ele not in count1:
            count1[ele] = 0
    imgname_to_mv = []
    for num in count1:
        if count1[num] < 250 and count[num] != 0:
            img_to_mv = []
            for cat in simdata:
                if cat[1] == num:
                    if count[num] + count1[num] <=250:
                        img_to_mv.append(cat[0])
                        imgname_to_mv = list(set(imgname_to_mv+img_to_mv))
                    else:
                        if len(img_to_mv) < 250-count1[num] :
                            img_to_mv.append(cat[0])
                            imgname_to_mv = list(set(imgname_to_mv+img_to_mv))
                else:
                    pass
    for imgname in imgname_to_mv:
        name = str(imgname) + ".png"
        shutil.copy(simpicpath+name,comdatapath)
    id_to_mv = list(k for k in data['images'] if k['id'] in imgname_to_mv)
    ann_to_mv = list(p for p in data['annotations'] if p['image_id'] in imgname_to_mv)
    for ele in id_to_mv:
        imgdata.append(ele)
    for ele in ann_to_mv:
        data2.append(ele)
    with open(outfilepath,'w',encoding='utf') as w:
        temp_dict = {'licenses':licdata, 'type':typedata, 'info':infodata, 'images':imgdata, 'annotations':data2,'categories':catdata}
        json.dump(temp_dict,w)
    w.close()
    ff.close()
    f.close()

                        


    
        


         





coverttrainjson()
print("训练注释文件转换完成.....")
covertvaljson()
print("验证注释文件转换完成.....")
print('开始进行模型转换>>>>>>>>>')
Cocopretrained()
print('模型转换完成.........')
# print('开始在多分类数据集中添加数据')
# Addsimpledata('/cache/train/train2.json','/cache/train/traincom.json','/cache/train/images_withoutrect/',
#             '/cache/train/train_images/','/cache/train/syntrain.json')
# print('添加数据集完成')