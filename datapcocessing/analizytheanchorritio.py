import pandas as pd
import seaborn as sns
import numpy as np
import json
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['figure.figsize'] = (10.0, 10.0)

ann_json = './complex_trash.json'
with open(ann_json,'r',encoding='utf-8') as f:
    ann=json.load(f)

#################################################################################################
#创建类别标签字典
category_dic=dict([(i['id'],i['name']) for i in ann['categories']])
counts_label=dict([(i['name'],0) for i in ann['categories']])
print(counts_label)
for i in ann['annotations']:
    counts_label[category_dic[i['category_id']]]+=1
print(counts_label)


box_w = []
box_h = []
box_wh = []
categorys_wh = [[] for j in range(204)]
for a in ann['annotations']:
    if a['category_id'] != 0:
        box_w.append(round(a['bbox'][2],2))
        box_h.append(round(a['bbox'][3],2))
        wh = round(a['bbox'][2]/a['bbox'][3],0)
        if wh <1 :
            wh = round(a['bbox'][3]/a['bbox'][2],0)
        box_wh.append(wh)

        categorys_wh[a['category_id']-1].append(wh)
# print(box_wh)
# print(categorys_wh)



box_wh_unique = list(set(box_wh))
# print(box_wh_unique)
box_wh_count=[box_wh.count(i) for i in box_wh_unique]
# print(box_wh_count)

# 绘图
wh_df = pd.DataFrame(box_wh_count,index=box_wh_unique,columns=['宽高比数量'])
wh_df.plot(kind='bar',color="#55aacc")
plt.show()