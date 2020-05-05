import os
import random 
import  shutil
import json
import re
import cv2
import glob
from PIL import Image



def Createdir(path):

    
    if not os.path.exists(path):
        os.mkdir(path)



#由于单类垃圾训练时读取png图片时老出错，所以将png图片转换成jpg格式的图片进行读取
def Pngtojpg(dirname_read,dirname_write):

    

    print("开始将PNG图片转换成jpg图片>>>>>>>>>")
    Createdir(dirname_write)

    names=os.listdir(dirname_read)
    count=0
    for name in names:
        img=Image.open(dirname_read+name)
        name=name.split(".")
        if name[-1] == "png":
            name[-1] = "jpg"
            name = str.join(".", name)
        #print(name)
        # r,g,b,a=img.split()              
        # img=Image.merge("RGB",(r,g,b))   
            to_save_path = dirname_write + name
            img.save(to_save_path)
            count+=1
        # print(" %s covert to %s" %(img, name))
        else:
            continue
    print("PNG图片已经转换成相应的JPG格式.......")


#-------------------------------------------------------------------------------------------------------
#-----------------------------------
#此部分是对多分类图片及注释文件的处理
#-----------------------------------
#从训练数据集中挑选出测试集数据，并放在另外一个文件夹中
def Dividimage(fileDir,tarDir):


    print("从验证集数中分开测试集数据>>>>>>>>")
    Createdir(tarDir)
    pathDir = os.listdir(fileDir)    #取图片的原始路径
    filenumber=len(pathDir)
    rate=0.167    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
    # print (sample)
    for name in sample:
            shutil.move(fileDir+name, tarDir+name)
    print("分出验证集数据完毕.......")


#从验证集数据集分开测试集数据，并将注释文件从验证集注释数据分开
def Dvidjson(filepath,tarpath,testpath):

    print("开始从验证注释文件中分开测试数据>>>>>>")
    with open(filepath,'r',encoding='utf-8') as f:
        temp_dict = {}
        result = []
        data = json.load(f)
        temp_dict['licenses'] = data['linceses']
        temp_dict['type'] = data['type']
        temp_dict['info'] = data['info']
        temp_dict['categories'] = data['categories']
        l = len(data['images'])
        img = data['images']
        testlist = os.listdir(testpath)
        testlength = len(testlist)
        for i in range(l):
            for test in testlist:
                if img[i]['file_name'] == 'valimages/' + test :
                    string = img[i]['file_name']
                    string1 = re.sub(r'valimages/|jpg','testimages',string,1)
                    img[i]['file_name'] = string1
                    result.append(img[i])
        temp_dict['images'] = result
        with open (tarpath,'w',encoding='UTF-8') as w:
            json.dump(temp_dict,w)
        w.close()

    f.close()
    print("从验证集注释文件中分开测试注释文件完成.......")

#针对注释文件中的file_name的名字与其所下载的路径的名字不同，所以这里定义一个更改名字的函数
def ChangeFileName(path,newname):


    for category in os.listdir(path):
        if os.path.isdir(os.path.join(path,category))==True:
            new_name = category.replace(category,newname)
            os.rename(os.path.join(path,category),os.path.join(path,new_name))
#这个函数可以整合到ChangeValjson文件中
#---------------------------------------------------------------------------------------------------------------------

#图片转换成jpg后更改json文件中不符合coco数据集规范的json文件,将图片png变成jpg格式的图片
#更改训练数据的json文件，将图片的file_name更改
def ChangeTrainjson(filepath,tarpath):
    

    print("训练集数据注释文件开始转换>>>>>>>>")
    temp_dict = {}
    result1 = []
    result2 = []
    f = open(filepath,'r',encoding='UTF-8')
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
        name= data1[i]['file_name']
        string = name
        # string[-3:] = 'jpg'
        # name1 = ''.join(string)
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
    with open (tarpath,'w',encoding='UTF-8') as w:
        json.dump(temp_dict,w)
    w.close()
    f.close()
    print("训练集数据文件转换完成.......")


def ChangeValjson(filepath,tarpath):


    print("验证集数据注释文件开始转换>>>>>>")
    temp_dict = {}
    result1 = []
    result2 = []
    f = open(filepath,'r',encoding='UTF-8')
# w = open('./traintest.json','w')
    data = json.load(f)
    # print(data)
    temp_dict['licenses'] = data['licenses']
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
        name = data1[i]['file_name']
        string = name
        # string[-3:] = 'jpg'
        # name1 = ''.join(string)
        string1 = re.sub(r"images_withoutrect/|png",'',string,1)
        img['file_name'] = string1
        result1.append(img)
    temp_dict['images'] = result1    
    # temp_dict['annotations'] = result2
    with open (tarpath,'w',encoding='UTF-8') as w:
        json.dump(temp_dict,w)
    w.close()
    f.close()
    print("验证集注释数据转换完成......")


def findtrainerrorpicture(traimgpath,annoimgpath,annotarpath):


    
    imglist = glob.glob(os.path.join(traimgpath,'*png'))
    l = len(imglist)
    print('总图片数量为：%d'%l)
    num = 0
    wrongpc = []
    temp_dict = {}
    for i in range(l):
        img = cv2.imread(imglist[i])
        index = random.randint(2,2000)
        img2 = imglist[index+1]
        img3 = imglist[index-1]
        if index <= (l-i-1):
            img1 = cv2.imread(imglist[index+i])
        else:
            img1 = cv2.imread(imglist[i-index])
        if type(img) != type(img1) and type(img) != type(img2) and type(img) != type(img3):
            # print('错误的图片是：',imglist[i])
            num += 1
            wrongpc.append(imglist[i])
            os.remove(imglist[i])
    print('错图片的数量为：%d'%num)
    print("错图片列表为：",wrongpc)
    print("删除错误图片后数据集中的图片数量为：%d"%(len(imglist)))
    with open(annoimgpath,'r',encoding='utf-8') as w:
        data = json.load(w)
        data1 = data['annotations']
        data2 = data['images']
        lll = len(data2)
        ll = len(data1)
        j = 0
        while j < ll:
            # print('上面的ll为：',ll)
            string = data1[j]['image_id']
            m = 0
            j += 1
            while m <= len(wrongpc)-1:
                if wrongpc[m].split('/cache/train/images_withoutrect/')[1] != str(string) + '.png' :
                    # print(wrongpc[m].split('/cache/train/images_withoutrect/')[1])
                    m += 1
                else:
                    data1.pop(j-1)
                    ll = len(data1)
                    j = j - 1
                    break
            
        
        p = 0
        print('原来的images的长度%d'%lll)
        while p < lll:
            string1 = data2[p]['id']
            q = 0
            p += 1
            while q < len(wrongpc):
                if wrongpc[q].split('/cache/train/images_withoutrect/')[1] != str(string1) + '.png':
                    q += 1
                    
                else:
                    data2.pop(p-1)
                    lll = len(data2)
                    p = p - 1
                    break
                
        print('删除完成后images的长度为：%d'%lll)
        temp_dict['annotations'] = data1
        temp_dict['licenses'] = data['licenses']
        temp_dict['info'] = data['info']
        temp_dict['type'] = data['type']
        temp_dict['categories'] = data['categories']
        temp_dict['images'] = data2
        with open(annotarpath,'w',encoding='utf-8') as ww:
            json.dump(temp_dict,ww)
        ww.close()
    w.close()


def findvalerrorpicture(traimgpath,annoimgpath,annotarpath):


    imglist = glob.glob(os.path.join(traimgpath,'*png'))
    l = len(imglist)
    print('总图片数量为：%d'%l)
    num = 0
    wrongpc = []
    temp_dict = {}
    for i in range(l):
        img = cv2.imread(imglist[i])
        index = random.randint(2,800)
        img2 = imglist[index+1]
        if index <= (l-i-1):
            img1 = cv2.imread(imglist[index+i])
        else:
            img1 = cv2.imread(imglist[i-index])
        if type(img) != type(img1) and type(img) != type(img2) :
            # print('错误的图片是：',imglist[i])
            num += 1
            wrongpc.append(imglist[i])
            os.remove(imglist[i])
    print('错图片的数量为：%d'%num)
    print("错图片列表为：",wrongpc)
    print("删除错误图片后数据集中的图片数量为：%d"%(len(imglist)))
    with open(annoimgpath,'r',encoding='utf-8') as w:
        data = json.load(w)
        data1 = data['images']
        ll = len(data1)
        p = 0
        print('原来的images的长度%d'%ll)
        while p < ll:
            string1 = data1[p]['id']
            q = 0
            p += 1
            while q < len(wrongpc):
                if wrongpc[q].split('/cache/val/images_withoutrect/')[1] != str(string1) + '.png':
                    q += 1
                    
                else:
                    data1.pop(p-1)
                    ll = len(data1)
                    p = p - 1
                    break
        print('删除完成后images的长度为：%d'%ll)
        temp_dict['licenses'] = data['licenses']
        temp_dict['info'] = data['info']
        temp_dict['type'] = data['type']
        temp_dict['categories'] = data['categories']
        temp_dict['images'] = data1
        with open(annotarpath,'w',encoding='utf-8') as ww:
            json.dump(temp_dict,ww)
        ww.close()
    w.close()





def Checkjson_dataset(imgpath,filepath,tarpath):


    imglist = os.listdir(imgpath)
    length = len(imglist)
    temp_dict = {}
    with open(filepath,'r',encoding='UTF-8'):
        data = json.load(filepath)
        data1 = data['images']
        data2 = data['annotations']
        wrongid = []
        l = len(data1)
        ll = len(data2)
        for i in range(l):
            string = data1[i]['id']
            k = 0
            while k < length:
                string1 = imglist[k].split('.')[0]
                if str(string) == string1 :
                    
                    break
                else:
                    k +=1
            if k > length:
                wrongid.append(string)
                data1.pop(i)
        for j in range(ll):
            string2 = data2[j]['image_id']
            f = 0
            while f < len(wrongid):
                if string2 == wrongid[f]:
                    data2.pop(j)
                else:
                    f += 1
        temp_dict['info'] = data['info']
        temp_dict['type'] = data['type']
        temp_dict['licenses'] = data['licenses']
        temp_dict['categories'] = data['categories']
        temp_dict['images'] = data1
        temp_dict['annotations'] = data2
        with open(tarpath,'w',encoding='utf-8') as w:
            json.dump(tarpath,w)
        w.close()
    f.close()



    

if __name__ == '__main__':
    
    # Pngtojpg("/cache/train/images_withoutrect/","/cache/train/images/")
    # Pngtojpg('./cache/val/images_withoutrect/','./cache/val/valimages/')
    ChangeTrainjson('/cache/train/train.json','/cache/train/train1.json')
    # ChangeValjson('/cache/val/val_open.json','/cache/val/val1.json')
    findtrainerrorpicture('/cache/train/images_withoutrect/','/cache/train/train1.json','/cache/train/train2.json')
    # findvalerrorpicture('/cache/val/images_withoutrect/','/cache/val/val1.json','/cache/val/val2.json')
    # Dividimage('./cache/val/valimages/','./cache/val/testimages/')
    # Dvidjson('./cache/val/val1.json','./cache/val/val2.json','./cache/val/testimages/')
