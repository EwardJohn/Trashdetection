import time
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def Getdata(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data1 = data['annotations']
        data2 = data['images']
        l = len(data1)
        print(l)
        ll = len(data2)
        result = []

        for i in range(l):
            string = data1[i]['bbox']
            # if string[2] <= 600 and string[3] <= 700:
            result.append(string[2:])
        return result
    f.close()

start = time.time()
result = Getdata('./complex_trash.json')
print(len(result))
clf = KMeans(n_clusters=10,init='k-means++',n_init=30,max_iter=5000,tol=0.00000001,algorithm='auto')
y_pred = clf.fit_predict(result)
x = [n[0] for n in result]
y = [n[1] for n in result]
'''这个是必须写的，相当于上面构造出来，配置好，下面这句调用，当然也可以写到上面去
fit方法对数据做training 并得到模型'''
# estimator.fit(dataSet)#聚类

# 下面是三个属性
'''把聚类的样本打标签'''
# labelPred = estimator.labels_
# print(labelPred)
'''显示聚类的质心'''
# centroids = estimator.cluster_centers_
# print(centroids)
# '''这个也可以看成损失，就是样本距其最近样本的平方总和'''
# inertia = estimator.inertia_
# print(inertia)
# kc = estimator.cluster_centers_
# y_kmeans = estimator.predict(result)
# print(np.shape(labelPred))
# x_list = y_list = []
# for ele in result:
#     x_list.append(ele[0])
#     y_list.append(ele[1])
# plt.figure('anchor')
# ax = plt.gca()
# ax.set_xlabel('x')
# ax.set_ylabel('y')
plt.title('anchor_scale')
plt.xlabel('anchor_width')
plt.ylabel('anchor_height')
# plt.legend(['Rank'])
plt.scatter(x, y, c=y_pred, s=20, alpha=0.5,marker='x')
plt.show()
end = time.time()
print(end-start)