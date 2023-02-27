

import math
import tifffile
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import skimage
import os
from PIL import Image
from skimage import measure,color

from glob import glob
import seaborn as sns
from tqdm import tqdm
from sklearn import linear_model  
import pandas as pd
import random
from scipy import stats
from scipy.stats import zscore
from sklearn.impute import KNNImputer
from numba import jit


# In[2]:


"""
please test version before use
"""
import numpy
import numba
print(numpy.__version__)
print(skimage.__version__)
numba.__version__


# In[100]:


"""

筛选细胞面积，
闭开运算分割细胞去除噪音

获得筛选后标签图片
在这种情况下，我们得到更多的单细胞，但是损失了细节


    开运算：先腐蚀后膨胀, 去除噪声，去除白色小点、空洞
    闭运算：先膨胀后腐蚀, 用来填充前景物体的小黑点
    形态学梯度：膨胀减去腐蚀, 可以得到前景物体的轮廓
    礼帽：原图减去开运算
    黑帽：闭运算减去原图
    
thresh=2 胞核
thresh=1 细胞核+细胞质

"""

def get_labeled_pred(pred_img, thresh=1): 
    pred_img=Image.open(pred_img)
    size=np.array(pred_img).shape
    
    ret, thresh_pred = cv.threshold(np.array(pred_img), thresh, 1, cv.THRESH_BINARY)
    #形态学调整
#     kernel = np.ones((4,4),np.uint8)
#     closing = cv.morphologyEx(thresh_pred, cv.MORPH_CLOSE, kernel)
#     kernel2 = np.ones((10,10),np.uint8)
#     thresh_pred = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel2)
    
    labels_objects = measure.label(thresh_pred, background=0, connectivity=1)

    properties = measure.regionprops(labels_objects)
    object_areas = []
    object_labels = []
    object_coords= []
    for prop in properties:
        object_areas.append(prop.area)
        object_labels.append(prop.label)
        object_coords.append(prop.coords)
    #limitmax=np.mean(object_areas) + (np.std(object_areas)*4)      #3
    limitmin=np.mean(object_areas) *1 #- (np.std(object_areas)*0.5)    #0.1
    valid_labels=[]
    for label1,area in enumerate(object_areas):
        if area>limitmin :   #  and area<limitmax
            valid_labels.append(label1)
    valid_coords=[]
    for label2,coord in enumerate(object_coords):
        for label1 in valid_labels:
            if label2==label1:
                valid_coords.append(object_coords[label2])
    imgnew=np.zeros((size))
    for k in valid_coords:
        for i in k:
            imgnew[i[0],i[1]]=1
    valid_labels_objects=measure.label(imgnew,background=0,connectivity=1)
        
    return valid_labels_objects


# In[9]:


"""
加入分水岭的细胞label
watershed added label
"""
def get_labeled_pred_watershed(pred_img, thresh=1): 
    img=cv.imread(pred_img)
    pred_img=Image.open(pred_img)
    gray = np.array(pred_img)# cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, gray = cv.threshold(gray, thresh, 1, cv.THRESH_BINARY)
#     plt.figure(figsize=(42,42))
#     plt.imshow(gray)
    ret, thresh = cv.threshold(gray,0,1,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    thresh=1-thresh
#     plt.figure(figsize=(3,3))
#     plt.imshow(thresh)
    # 噪声去除
    kernel = np.ones((5,5),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    # 确定背景区域
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    # 寻找前景区域
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.1*dist_transform.max(),1,0)
    # 找到未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
#     plt.figure(figsize=(8,8))
#     plt.imshow(unknown)
    # 类别标记
    ret, markers = cv.connectedComponents(sure_fg)
#     plt.figure(figsize=(7,7))
#     plt.imshow(markers+1)
    # 为所有的标记加1，保证背景是0而不是1
    markers = markers+1
    # 现在让所有的未知区域为0
    markers[unknown==1] = 0
    markers = cv.watershed(img,markers) 
    gray[markers == -1] = [0]
    
    
    size=np.array(pred_img).shape
    ret, thresh_pred = cv.threshold(np.array(gray), 0, 1, cv.THRESH_BINARY)
    #形态学调整
#     kernel = np.ones((4,4),np.uint8)
#     closing = cv.morphologyEx(thresh_pred, cv.MORPH_CLOSE, kernel)
#     kernel2 = np.ones((10,10),np.uint8)
#     thresh_pred = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel2)
#    plt.imshow(thresh_pred)
    labels_objects = measure.label(thresh_pred, background=0, connectivity=1)

    properties = measure.regionprops(labels_objects)
    object_areas = []
    object_labels = []
    object_coords= []
    for prop in properties:
        object_areas.append(prop.area)
        object_labels.append(prop.label)
        object_coords.append(prop.coords)
    limitmax=np.mean(object_areas) + (np.std(object_areas)*3)      #3
    limitmin=np.mean(object_areas) *0.7 #- (np.std(object_areas)*0.5)    #0.1
    valid_labels=[]
    for label1,area in enumerate(object_areas):
        if area<limitmax and area>limitmin:
            valid_labels.append(label1)
    valid_coords=[]
    for label2,coord in enumerate(object_coords):
        for label1 in valid_labels:
            if label2==label1:
                valid_coords.append(object_coords[label2])
    imgnew=np.zeros((size))
    for k in valid_coords:
        for i in k:
            imgnew[i[0],i[1]]=1
    valid_labels_objects=measure.label(imgnew,background=0,connectivity=1)
        
    return valid_labels_objects


# In[ ]:


"""
modulation of cell screen principle
载入单张图片测试筛选标准
寻找标准 区域范围
"""

singleimgdir1=r"C:\Users\Mengxi\Box\Data\SingleCell\20220514_stress_test_repeat\stress_repeat_test_14_5_2022\sample_1_1\Pos0\Unet++_of_434_Norm_withmodel-0.707_orig0413_pth\img_000000000_438-542-4_6_000.png"    
testimg1=singleimgdir1
Image.open(singleimgdir1)
plt.figure(figsize=(8,8))
#plt.imshow(testimg1)

labeled_pred_test=get_labeled_pred(testimg1,thresh=1)
labeled_pred_test_2=get_labeled_pred(testimg1,thresh=2)
plt.figure(figsize=(12,12))
plt.imshow(labeled_pred_test)
plt.figure(figsize=(42,42))
plt.imshow(labeled_pred_test_2)
print(np.max(labeled_pred_test_2))


# In[11]:


"""
在label删选过的基础上获取信息
获取的信息注意cell——0对应标签图片的label——1
0 label
1 面积 area
2 周长 perimeters
3 重心 cetroids
4 bbox bbox
5 坐标 coords
6 离心 eccentricityes
7 image
"""
def get_labeled_info(labeled_pred):
    properties=measure.regionprops(labeled_pred)
    object_areas = []
    object_perimeters = []  # 周长
    object_centroids = []  # 重心点
    object_bboxs = []
    object_coords = []  # 区域内像素点坐标
    object_labels = []
    object_eccentricities=[]
    object_images=[]
    valid_para_of_1image=[]
    for prop in properties:
        object_areas.append(prop.area)
        object_perimeters.append(prop.perimeter)
        object_centroids.append(prop.centroid)
        object_bboxs.append(prop.bbox)
        object_coords.append(prop.coords)
        object_labels.append(prop.label)
        object_eccentricities.append(prop.eccentricity)
        object_images.append(prop.image)
    ziped_para_of_image = list(zip(object_labels,      # 0
                                   object_areas,      # 1 面积
                                   object_perimeters,  # 2周长
                                   object_centroids,  # 3重心点
                                   object_bboxs,
                                   object_coords,  # 5区域内像素点坐标                                   
                                   object_eccentricities,
                                  object_images))        
    return ziped_para_of_image


# In[12]:


"""
获得纯细胞质区域（去除细胞核）
"""

def get_labeled_pred_only_cyto(pred_img): 
    img=cv.imread(pred_img)
    pred_img=Image.open(pred_img)
    
    size=np.array(pred_img).shape
    thresh_nuclei=2
    thresh_cyto=1
    ret, thresh_pred_neclei = cv.threshold(np.array(pred_img), thresh_nuclei, 1, cv.THRESH_BINARY)
    ret, thresh_pred_cyto = cv.threshold(np.array(pred_img), thresh_cyto, 1, cv.THRESH_BINARY)
    thresh_pred=thresh_pred_cyto-thresh_pred_neclei
        
    size=np.array(pred_img).shape
    #ret, thresh_pred = cv.threshold(np.array(gray), 0, 1, cv.THRESH_BINARY)
    
    labels_objects = measure.label(thresh_pred, background=0, connectivity=1)

    properties = measure.regionprops(labels_objects)
    object_areas = []
    object_labels = []
    object_coords= []
    for prop in properties:
        object_areas.append(prop.area)
        object_labels.append(prop.label)
        object_coords.append(prop.coords)
    limitmax=np.mean(object_areas) + (np.std(object_areas)*3)
    limitmin=np.mean(object_areas) *0.7 #- (np.std(object_areas)*0.3)
    valid_labels=[]
    for label1,area in enumerate(object_areas):
        #if area<2800 and area>600
        #if area<3500 and area>500:
        if area<limitmax and area>limitmin:
            valid_labels.append(label1)
    valid_coords=[]
    for label2,coord in enumerate(object_coords):
        for label1 in valid_labels:
            if label2==label1:
                valid_coords.append(object_coords[label2])
    imgnew=np.zeros((size))
    for k in valid_coords:
        for i in k:
            imgnew[i[0],i[1]]=1
    valid_labels_objects=measure.label(imgnew,background=0,connectivity=1)
        
    return valid_labels_objects


# In[13]:


"""
加入watershed
获得纯细胞质区域（去除细胞核）

"""

def get_labeled_pred_only_cyto_watershed(pred_img): 
    img=cv.imread(pred_img)
    pred_img=Image.open(pred_img)
    
    size=np.array(pred_img).shape
    thresh_nuclei=2
    thresh_cyto=1
    ret, thresh_pred_neclei = cv.threshold(np.array(pred_img), thresh_nuclei, 1, cv.THRESH_BINARY)
    
    ret, thresh_pred_cyto = cv.threshold(np.array(pred_img), thresh_cyto, 1, cv.THRESH_BINARY)
    
    gray=thresh_pred_cyto
    #形态学调整
#     kernel = np.ones((4,4),np.uint8)
#     closing = cv.morphologyEx(thresh_pred, cv.MORPH_CLOSE, kernel)
#     kernel2 = np.ones((10,10),np.uint8)
#     thresh_pred = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel2)
    ret, thresh = cv.threshold(gray,0,1,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    thresh=1-thresh
#     plt.figure(figsize=(3,3))
#     plt.imshow(thresh)
    # 噪声去除
    kernel = np.ones((5,5),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    # 确定背景区域
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    # 寻找前景区域
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.1*dist_transform.max(),1,0)
    # 找到未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
#     plt.figure(figsize=(8,8))
#     plt.imshow(unknown)
    # 类别标记
    ret, markers = cv.connectedComponents(sure_fg)
#     plt.figure(figsize=(7,7))
#     plt.imshow(markers+1)
    # 为所有的标记加1，保证背景是0而不是1
    markers = markers+1
    # 现在让所有的未知区域为0
    markers[unknown==1] = 0
    markers = cv.watershed(img,markers) 
    thresh_pred_cyto[markers == -1] = [0]
    
    thresh_pred=thresh_pred_cyto-thresh_pred_neclei
    #形态学调整
#     kernel = np.ones((4,4),np.uint8)
#     closing = cv.morphologyEx(thresh_pred, cv.MORPH_CLOSE, kernel)
#     kernel2 = np.ones((10,10),np.uint8)
#     thresh_pred = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel2)
    
    
    labels_objects = measure.label(thresh_pred, background=0, connectivity=1)

    properties = measure.regionprops(labels_objects)
    object_areas = []
    object_labels = []
    object_coords= []
    for prop in properties:
        object_areas.append(prop.area)
        object_labels.append(prop.label)
        object_coords.append(prop.coords)
    limitmax=np.mean(object_areas) + (np.std(object_areas)*3)
    limitmin=np.mean(object_areas) - (np.std(object_areas)*0.3)
    #print(limitmin)
    valid_labels=[]
    for label1,area in enumerate(object_areas):
        #if area<2800 and area>600
        #if area<3500 and area>500:
        if area<limitmax and area>limitmin:
            valid_labels.append(label1)
    valid_coords=[]
    for label2,coord in enumerate(object_coords):
        for label1 in valid_labels:
            if label2==label1:
                valid_coords.append(object_coords[label2])
    imgnew=np.zeros((size))
    for k in valid_coords:
        for i in k:
            imgnew[i[0],i[1]]=1
    valid_labels_objects=measure.label(imgnew,background=0,connectivity=1)
        
    return valid_labels_objects


# In[107]:


"""
载入单张图片测试筛选标准
寻找标准 区域范围
"""
singleimgdir1=r"C:\Users\Mengxi\Box\Data\SingleCell\20220514_stress_test_repeat\stress_repeat_test_14_5_2022\sample_3_1\Pos0\Unet++_of_434_Norm_withmodel-0.707_orig0413_pth\img_000000000_438-542-4_6_000.png"    
testimg1=Image.open(singleimgdir1)
plt.figure(figsize=(8,8))
plt.imshow(testimg1)
plt.figure(figsize=(12,12))
labeled_pred_test_cyto=get_labeled_pred_only_cyto_watershed(singleimgdir1)
plt.imshow(labeled_pred_test_cyto)
infoff=get_labeled_info(labeled_pred_test_cyto)
arealist=[]
for i in infoff:
    arealist.append(i[1])

print(np.min(arealist))


# In[17]:


"""
测试最大最小面积，最小面积至少大于10 不可以小于0
"""
area=[]
for i in get_labeled_info(labeled_pred_test):
    area.append(i[1])
print(np.max(area))
print(np.min(area))


# In[18]:


def make_small_img(target_cell_num, labeled_pred_img, pred_info, size):
    sizeofimg=np.array(labeled_pred_img).shape
    if target_cell_num is not None:
        x = int(pred_info[target_cell_num][3][0])
        y = int(pred_info[target_cell_num][3][1])
        small_img=labeled_pred_img[np.max([0,x-size]):np.min([x+size,sizeofimg[0]]),
                        np.max([0,y-size]):np.min([y+size,sizeofimg[1]])]
        return small_img
    else:
        return None


# In[19]:


size=np.array(testimg1).shape

def represent_single_cell_from(pred_info, num):
    
    if num is not None: 
        coords = pred_info[num][5]
        ohhh_img = np.zeros((size))
        for i in coords:
            x = int(i[0])
            y = int(i[1])
            ohhh_img[x, y] = 1
        plt.imshow(ohhh_img)
    else:
        print("no matching objects,num is None")


# In[20]:


def get_img_binary(pred,thresh):
    ret,thresh_pred_1=cv.threshold(pred,thresh,1,cv.THRESH_BINARY)
    return thresh_pred_1


# In[21]:


"""

获取一个细胞（第一个图片中的细胞）在单个目标图片中的相应序号
目标在后
要求输入 单个图片 和 单个信息！！
"""
import math
def find_color_label(num,pred_info_ori,labeled_pred_tar,pred_info_tar,size=200,disThresh=6):    
    # 以第一张图的重心为中心，划出目标图局部区域
    sizeofimg=np.array(labeled_pred_tar).shape
    if num is None:
        print("target cell num is None")
    else:    
        if num>len(pred_info_ori):
            print(f"{num} is out of cell range {len(pred_info_ori)}")
        else:
            #print(pred_info_ori[num][3][0])
            x = int(pred_info_ori[num][3][0])
            y = int(pred_info_ori[num][3][1])
            small_img=labeled_pred_tar[np.max([0,x-size]):np.min([x+size,sizeofimg[0]]),
                            np.max([0,y-size]):np.min([y+size,sizeofimg[1]])]      
            simg2=small_img
            if simg2 is not None:
                #plt.imshow(simg2)
                cell_num=np.unique(simg2)[1:]-1

                distances=[]
                for label in cell_num:
                    x=int(pred_info_tar[label][3][0])
                    y=int(pred_info_tar[label][3][1])
                    distance=math.sqrt(((int(pred_info_ori[num][3][0])-x)**2 + (int(pred_info_ori[num][3][1])-y)**2))

                    distances.append(distance)
                #print(distances)
                target=None
                if np.size(distances)==0:
                    target=None
                else:
                    if np.min(distances)<disThresh:
                        target=cell_num[np.argmin(distances)]

                return target


# In[22]:
"""
加入了距离 面积 周长的 cell tracking
hungarian algorithm
cell cost
"""
import math


def find_color_label(num, pred_info_ori, labeled_pred_tar, pred_info_tar, size=200, disThresh=6):
    # 以第一张图的重心为中心，划出目标图局部区域
    sizeofimg = np.array(labeled_pred_tar).shape
    if num is None:
        print("target cell num is None")
    else:
        if num > len(pred_info_ori):
            print(f"{num} is out of cell range {len(pred_info_ori)}")
        else:
            # print(pred_info_ori[num][3][0]) # 画出一个200*200的小图减少计算量
            x_ori = int(pred_info_ori[num][3][0])
            y_ori = int(pred_info_ori[num][3][1])
            small_img = labeled_pred_tar[np.max([0, x_ori - size]):np.min([x_ori + size, sizeofimg[0]]),
                        np.max([0, y_ori - size]):np.min([y_ori + size, sizeofimg[1]])]
            simg2 = small_img
            if simg2 is not None:
                # plt.imshow(simg2)
                cell_num = np.unique(simg2)[1:] - 1

                distances = []
                for label in cell_num:
                    x_tar = int(pred_info_tar[label][3][0])
                    y_tar = int(pred_info_tar[label][3][1])
                    distance1 = math.sqrt((x_ori - x_tar) ** 2 + (y_ori - y_tar) ** 2)  # 距离

                    distance2 = np.abs(pred_info_tar[label][1] - pred_info_ori[num][1])  # 面积

                    distance3 = np.abs(pred_info_tar[label][2] - pred_info_ori[num][2])  # 周长

                    distance = distance1 * 2 + distance2 * 0.5 + distance3 * 0.25
                    distances.append(distance)
                # print(distances)
                target = None
                if np.size(distances) == 0:
                    target = None
                else:
                    #if np.min(distances) < disThresh:
                    target = cell_num[np.argmin(distances)]

                return target


"""
hungarian algorithm
cell cost
"""
import math

from scipy.optimize import linear_sum_assignment

def find_color_label(pred_info_ori, labeled_pred_tar, pred_info_tar, size=200):

    RR=len(pred_info_ori)
    CC=len(pred_info_tar)
    cost=np.zeros((RR,CC))
    for num in range(RR):
        for label in range(np.CC):
            x_ori = int(pred_info_ori[num][3][0])
            y_ori = int(pred_info_ori[num][3][1])
            x_tar = int(pred_info_tar[label][3][0])
            y_tar = int(pred_info_tar[label][3][1])

            try:
                distance1 = math.sqrt((x_ori - x_tar) ** 2 + (y_ori - y_tar) ** 2)  # 距离

                distance2 = np.abs(pred_info_tar[label][1] - pred_info_ori[num][1])  # 面积

                distance3 = np.abs(pred_info_tar[label][2] - pred_info_ori[num][2])  # 周长

                distance = np.float(distance1 * 2 + distance2 * 0.5 + distance3 * 0.25)

                cost[num][label]=distance
            except:
                pass

    cost = (0.5) * cost
    # Using Hungarian Algorithm assign the correct detected measurements
    # to predicted tracks
    assignment = []
    for _ in range(RR):
        assignment.append(-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    for i in range(len(row_ind)):
        assignment[row_ind[i]] = col_ind[i]

    return assignment









"""
tracking cell from first to each
"""
def get_same_cell_list_first_to_last(num,labeled_preds,preds_info):
    same_cell_list=[]
    for i,img in enumerate(labeled_preds):
        
        same_cell=find_color_label(num,preds_info[0],labeled_preds[i],preds_info[i])
        same_cell_list.append(same_cell)
    return same_cell_list


# In[23]:


"""
tracking cell  frame by frame
is better than tracking one cell from first to last？
"""
def track_cell_framebyframe(num_of_cell,labeled_preds_1,preds_info_1):
    
    same_cell_list1=[num_of_cell] #no_none_
    same_cell_list2=[num_of_cell] #none_included_
    validIlist=[0]
    for i,img in enumerate(labeled_preds_1):
        if i<len(labeled_preds_1)-1:
            if same_cell_list2[-1] is not None:
                same_cell=find_color_label(same_cell_list2[-1],preds_info_1[i],labeled_preds_1[i+1],preds_info_1[i+1],size=200,disThresh=6)
#                 same_cell_list2.append(same_cell)
#                 if same_cell is not None:
#                     same_cell_list1.append(same_cell)
#                     validIlist.append(i)
            else:
                #same_cell=find_color_label(num_of_cell,preds_info_1[0],labeled_preds_1[i+1],preds_info_1[i+1],size=200,disThresh=5)
                same_cell=find_color_label(same_cell_list1[-1],preds_info_1[validIlist[-1]],labeled_preds_1[i+1],preds_info_1[i+1],size=100,disThresh=6)
            same_cell_list2.append(same_cell)  
            if same_cell is not None:
                same_cell_list1.append(same_cell)
                validIlist.append(i+1)
#     print(same_cell_list2)
#     print(validIlist)
    return same_cell_list2


# In[24]:


"""

用同一套prediction计算两个相应的FRET
相减 获得细胞亮度

需要注意和修改：是谁比谁，
always
短的比长的
FRET=x/fret
乳酸浓度与1/fret正相关
短的比长的，做斜率不需倒置

"""

def calculate_brightness(num,target_imgfret1,target_imgfret2,one_pred_info,one_labled_pred):
    if num>len(one_pred_info):
        print(f"{num}, is out of image cell range {len(one_pred_info)}")
    else:
        box=one_pred_info[num][4]    

        onecellmask=one_pred_info[num][7]
        onecellarea=one_pred_info[num][1]

        small_target_pred=one_labled_pred[box[0]:box[2],box[1]:box[3]]
        maskofallcell=get_img_binary(np.float64(small_target_pred),0)
        bg_reversmask=[[1-i for i in j] for j in maskofallcell]
        bg_area=np.sum(bg_reversmask)

        bright1=[]
        bright1res=[]
        bright2=[]
        bright2res=[]
        pppsjfk=target_imgfret1.copy()
        pppsjfk2=target_imgfret2.copy()

        boxinfret1=target_imgfret1[box[0]:box[2],box[1]:box[3]]
        validfret1=boxinfret1*onecellmask
        outsiderfret1=boxinfret1*bg_reversmask

        boxinfret2=target_imgfret2[box[0]:box[2],box[1]:box[3]]
        validfret2=boxinfret2*onecellmask
        outsiderfret2=boxinfret2*bg_reversmask
        f1=(np.sum(validfret1)/onecellarea)-(np.sum(outsiderfret1)/bg_area)
        f2=(np.sum(validfret2)/onecellarea)-(np.sum(outsiderfret2)/bg_area)
        return(f2/f1)



# In[218]:


"""
一个超级计算公式
自己写晕了都

所有计算都在这里了
"""

def smooth_list(li):
    smooth_list=[]
    for i in range(len(li)):
        k=np.median(li[i:i+5])
        smooth_list.append(k)
    return smooth_list

def fillNA_smooth_data(randN):
    randN=pd.DataFrame(randN)
    randN[randN==None]=np.nan
    randN[np.isinf(randN)]=np.nan
    if randN.shape[1]!=0:
        imputer = KNNImputer(n_neighbors=5)
        randN=imputer.fit_transform(randN)

#         randN=randN*(np.abs(stats.zscore(randN)) < 1.62)
#         randN[randN==0]=np.nan
#         imputer = KNNImputer(n_neighbors=5)
#         randN=imputer.fit_transform(randN)
        randN=randN[np.logical_not(np.isnan(np.array(randN)))]
        randN=np.array(smooth_list(list(randN)))
    return randN

def get_all_parlists_on_agentAddTimelist_of_One_exp(labeled_preds,preds_info,image_FRET1,image_FRET2,timelist,agent_last_frame=50,smoofig=False,celldict=False):
    print("即将开始此程序最大最慢的运算，开始读条：")
    #agent_last_frame 加一个试剂持续多少个frame？？
    all_cell_dict={}
    all_cell_info_list=[]
    valid_frame_more85_celllist=[]
    allcellfrets=[]
    allcellareas=[]
    allcellperimeters=[]
    allcelleccents=[]
    all_cell_valid_slops_on_timelist=[]
    all_cell_cellnum_valid_slops_on_time_list=[]
    model = linear_model.LinearRegression()
    for llabel,i in enumerate(tqdm(range(len(preds_info[0])))):
    #for i in tqdm(range(least_cell_numbers(labeled_preds))):

        same_cell_list=track_cell_framebyframe(i,labeled_preds,preds_info)
        #same_cell_list=get_same_cell_list_first_to_last(num,labeled_preds,preds_info)
        how_many_valida_frames_of_one_cell=sum(x is not None for x in same_cell_list)
        # 只计算那些能够在85%以上图片中找得到的细胞：
        if how_many_valida_frames_of_one_cell > 0.85*(len(image_FRET1)):
            valid_frame_more85_celllist.append(llabel)
            sequence_of_imgs=range(len(image_FRET1))
            zipped=list(zip(sequence_of_imgs,same_cell_list))

            onecellfretlist=[]
            onecellarealist=[]
            onecellperimeterlist=[]
            onecelleccentricitieslist=[]

            onecellfretlist_after_agent=[]
            onecellarealist_after_agent=[]
            onecellperimeterlist_after_agent=[]
            onecelleccentricitieslist_after_agent=[]

    #       if how_many_valida_images > 0.9 * len(image_FRET1):
            for ll,(v,k) in enumerate(zipped):
                if k is None:
                    onecellfretlist.append(np.nan)
                    onecellarealist.append(np.nan)
                    onecellperimeterlist.append(np.nan)
                    onecelleccentricitieslist.append(np.nan)
                else:
                    brightnessRatio=calculate_brightness(k,image_FRET1[v],image_FRET2[v],preds_info[v],labeled_preds[v])
                    area=preds_info[v][k][1]
                    perimeter=preds_info[v][k][2]
                    eccentricity=preds_info[v][k][6]

                    onecellfretlist.append(brightnessRatio)
                    onecellarealist.append(area)
                    onecellperimeterlist.append(perimeter)
                    onecelleccentricitieslist.append(eccentricity)
            one_cell_info_list=list(zip(onecellfretlist,onecellarealist,onecellperimeterlist,onecelleccentricitieslist))
            all_cell_info_list.append(one_cell_info_list)
            #print("单个细胞合并四个信息形成的ziplist:",np.array(one_cell_info_list).shape)

            for add_agent_time in timelist:

                onecellfretlist_after_agent.append(onecellfretlist[add_agent_time:(add_agent_time+agent_last_frame)])
                onecellarealist_after_agent.append(onecellarealist[add_agent_time:(add_agent_time+agent_last_frame)])
                onecellperimeterlist_after_agent.append(onecellperimeterlist[add_agent_time:(add_agent_time+agent_last_frame)])
                onecelleccentricitieslist_after_agent.append(onecelleccentricitieslist[add_agent_time:(add_agent_time+agent_last_frame)])
            #print("一个细胞的时间段信息",np.array(onecellfretlist_after_agent).shape)

            onecellslops_of_timelist=[]
            onecellvalidslopsontimelistcellnum=[]
            for whichtimelist,timeslot_ofFRET in enumerate(onecellfretlist_after_agent):
                if sum(ifret != np.nan for ifret in timeslot_ofFRET)>agent_last_frame*0.7 and pd.DataFrame(timeslot_ofFRET).shape[1]!=0:         
                    timeslot_ofFRET=fillNA_smooth_data(timeslot_ofFRET)
                a=timeslot_ofFRET
                #if a.shape[1] !=0:
                a=np.array(a)
                x=range(len(a))
                x=np.reshape(x,(-1,1))
                y=np.array(a)
                if x.shape[0]!=0:
                    model.fit(x,y)
                    slop=model.coef_
                    if model.score(x,y)>0.7:
                        onecellslops_of_timelist.append(slop/0.01221)
                        slopcode=llabel,whichtimelist
                        onecellvalidslopsontimelistcellnum.append(slopcode)

            all_cell_valid_slops_on_timelist.append(onecellslops_of_timelist)
            all_cell_cellnum_valid_slops_on_time_list.append(onecellvalidslopsontimelistcellnum)

            allcellfrets.append(fillNA_smooth_data(onecellfretlist))#fillNA_smooth_data(
            allcellareas.append(fillNA_smooth_data(onecellarealist))
            allcellperimeters.append(fillNA_smooth_data(onecellperimeterlist))
            allcelleccents.append(fillNA_smooth_data(onecelleccentricitieslist))
    if smoofig:
        x1=range(len(allcellfrets))
        x2=range(len(allcellareas))
        x3=range(len(allcellperimeters))
        x4=range(len(allcelleccents))
        plt.figure(figsize=(6,12))
        plt.subplot(4,1,1)
        plt.title("CFP/FRET")
        plt.plot(x1,allcellfrets)
        plt.subplot(4,1,2)
        plt.title("AREA")
        plt.plot(x2,allcellareas)
        plt.subplot(4,1,3)
        plt.title("Perimeter")
        plt.plot(x3,allcellperimeters)
        plt.subplot(4,1,4)
        plt.title("eccentricities")
        plt.plot(x4,allcelleccents)

    print("所有合格细胞合并四个信息形成的ziplist(细胞序号，frame数，信息数):",np.array(all_cell_info_list).shape)

    print("所有细胞合格的slop:",all_cell_valid_slops_on_timelist)
    print("所有细胞合格的slop格式:",np.array(all_cell_valid_slops_on_timelist).shape)
    print("所有细胞合格的slop所对应的细胞序号以及对应第几次加试剂:",all_cell_cellnum_valid_slops_on_time_list)
    all_cell_dict=dict(zip(valid_frame_more85_celllist,all_cell_info_list))
    print("合格细胞号列表",valid_frame_more85_celllist)
    print("总细胞的数量：",len(preds_info[0]),"合格细胞的数量细胞数（more than 85% frame trackable），也就是获得的fret、面积、周长、离心率结果的个数：",len(valid_frame_more85_celllist),"\n" ,"有效细胞∩有效追踪比例：",len(valid_frame_more85_celllist)/len(preds_info[0]))
    print("数据格式：","\n", f" fret:{np.array(allcellfrets).shape}","\n",f"area:{np.array(allcellareas).shape}","\n",f"peri:{np.array(allcellareas).shape} ")
    if celldict:
        return all_cell_dict,all_cell_valid_slops_on_timelist,allcellfrets,allcellareas,allcellperimeters,allcelleccents
    else:
        return all_cell_valid_slops_on_timelist,allcellfrets,allcellareas,allcellperimeters,allcelleccents


# In[152]:


"""
单个细胞测试版
"""
def single_cell_fret_test0(num,labeled_preds,preds_info,image_FRET1,image_FRET2,timelist,agent_last_frame=50):   
#     valid_frame_more85_celllist=[]
    model = linear_model.LinearRegression()
    same_cell_list=track_cell_framebyframe(num,labeled_preds,preds_info)
    #same_cell_list=get_same_cell_list_first_to_last(num,labeled_preds,preds_info)
    how_many_valida_frames_of_one_cell=sum(x is not None for x in same_cell_list)
    # 只计算那些能够在85%以上图片中找得到的细胞：
    if how_many_valida_frames_of_one_cell > 0.85*(len(image_FRET1)):
#        valid_frame_more85_celllist.append(llabel)
        sequence_of_imgs=range(len(image_FRET1))
        zipped=list(zip(sequence_of_imgs,same_cell_list))

        onecellfretlist=[]
        onecellarealist=[]
        onecellperimeterlist=[]
        onecelleccentricitieslist=[]

        onecellfretlist_after_agent=[]
        onecellarealist_after_agent=[]
        onecellperimeterlist_after_agent=[]
        onecelleccentricitieslist_after_agent=[]

#       if how_many_valida_images > 0.9 * len(image_FRET1):
        for ll,(v,k) in enumerate(zipped):
            if k is None:
                onecellfretlist.append(np.nan)
                onecellarealist.append(np.nan)
                onecellperimeterlist.append(np.nan)
                onecelleccentricitieslist.append(np.nan)
            else:
                brightnessRatio=calculate_brightness(k,image_FRET1[v],image_FRET2[v],preds_info[v],labeled_preds[v])
                area=preds_info[v][k][1]
                perimeter=preds_info[v][k][2]
                eccentricity=preds_info[v][k][6]

                onecellfretlist.append(brightnessRatio)
                onecellarealist.append(area)
                onecellperimeterlist.append(perimeter)
                onecelleccentricitieslist.append(eccentricity)
        one_cell_info_list=list(zip(onecellfretlist,onecellarealist,onecellperimeterlist,onecelleccentricitieslist))
#        all_cell_info_list.append(one_cell_info_list)
        #print("单个细胞合并四个信息形成的ziplist:",np.array(one_cell_info_list).shape)

        for add_agent_time in timelist:

            onecellfretlist_after_agent.append(onecellfretlist[add_agent_time:(add_agent_time+agent_last_frame)])
            onecellarealist_after_agent.append(onecellarealist[add_agent_time:(add_agent_time+agent_last_frame)])
            onecellperimeterlist_after_agent.append(onecellperimeterlist[add_agent_time:(add_agent_time+agent_last_frame)])
            onecelleccentricitieslist_after_agent.append(onecelleccentricitieslist[add_agent_time:(add_agent_time+agent_last_frame)])
        print("一个细胞的时间段信息",np.array(onecellfretlist_after_agent).shape)

        onecellslops_of_timelist=[]
        onecellvalidslopsontimelistcellnum=[]
        for whichtimelist,timeslot_ofFRET in enumerate(onecellfretlist_after_agent):
            if sum(ifret != np.nan for ifret in timeslot_ofFRET)>agent_last_frame*0.5:
                timeslot_ofFRET=fillNA_smooth_data(timeslot_ofFRET)
            a=timeslot_ofFRET
            #if a.shape[1] !=0:
            a=np.array(a)
            x=range(len(a))
            x=np.reshape(x,(-1,1))
            y=np.array(a)
            model.fit(x,y)
            slop=model.coef_
            if model.score(x,y)>0.4:
                onecellslops_of_timelist.append(slop/0.01221)
                slopcode=whichtimelist
                onecellvalidslopsontimelistcellnum.append(slopcode)
            else:
                onecellslops_of_timelist.append(np.nan)
        return onecellslops_of_timelist,onecellvalidslopsontimelistcellnum,fillNA_smooth_data(onecellfretlist)
    
    
timelist_t1=[30,90,180]
timelist_t=[90]
onecellslops_of_timelist,onecellvalidslopsontimelistcellnum,smoothonecellfretlist=single_cell_fret_test0(5,*exp1,timelist_t1,agent_last_frame=40)   


# In[26]:


"""
with watershed
获得筛选过的labeled pred list
同步获得他们的图片信息
mode= 1 cyto+nuclei
mode= 2 only nuclei
mode= 3 only cyto
"""

from tqdm import tqdm

def get_labeled_preds_and_info(pred_list,mode):
    
    labeled_preds = []
    preds_info = []
    if mode==1:
        for pred in tqdm(pred_list):
            labeled_pred_cell = get_labeled_pred_watershed(pred, thresh=1)
            labeled_preds.append(labeled_pred_cell)
            preds_info.append(get_labeled_info(labeled_pred_cell))
    print("seg result number:",len(labeled_preds), len(preds_info))
    if mode==2:
        for pred in tqdm(pred_list):
            labeled_pred_cell = get_labeled_pred_watershed(pred, thresh=2)
            labeled_preds.append(labeled_pred_cell)
            preds_info.append(get_labeled_info(labeled_pred_cell))
    if mode==3:
        for pred in tqdm(pred_list):
            labeled_pred_cell = get_labeled_pred_only_cyto_watershed(pred)
            labeled_preds.append(labeled_pred_cell)
            preds_info.append(get_labeled_info(labeled_pred_cell))
    return labeled_preds,preds_info


# In[27]:


"""
no watershed
获得筛选过的labeled pred list
同步获得他们的图片信息
mode= 1 cyto+nuclei
mode= 2 only nuclei
mode= 3 only cyto
"""

from tqdm import tqdm

def get_labeled_preds_and_info_no_watershed(pred_list,mode):
    
    labeled_preds = []
    preds_info = []
    if mode==1:
        for pred in tqdm(pred_list):
            labeled_pred_cell = get_labeled_pred(pred, thresh=1)
            labeled_preds.append(labeled_pred_cell)
            preds_info.append(get_labeled_info(labeled_pred_cell))

    if mode==2:
        for pred in tqdm(pred_list):
            labeled_pred_cell = get_labeled_pred(pred, thresh=2)
            labeled_preds.append(labeled_pred_cell)
            preds_info.append(get_labeled_info(labeled_pred_cell))
    if mode==3:
        for pred in tqdm(pred_list):
            labeled_pred_cell = get_labeled_pred_only_cyto(pred)
            labeled_preds.append(labeled_pred_cell)
            preds_info.append(get_labeled_info(labeled_pred_cell))
    return labeled_preds,preds_info


# In[28]:


def dataloader_pred_fret1_fret2(root_dir, seg_folder_name, extend1, extend2):

    pred_FRET1_path = glob(os.path.join(root_dir, seg_folder_name, '*' + extend1 + ".png"))
    FRET1_path = glob(os.path.join(root_dir, '*' + extend1 + ".tif"))
    FRET2_path = glob(os.path.join(root_dir, '*' + extend2 + ".tif"))
    pred_FRET1 = pred_FRET1_path #[Image.openg(img) for img in tqdm(pred_FRET1_path)]
    image_FRET1 = [tifffile.imread(img) for img in tqdm(FRET1_path)]
    image_FRET2 = [tifffile.imread(img) for img in tqdm(FRET2_path)]
    print(
        f'预测fret1图片数：{len(pred_FRET1)},原始fret1图片数：{len(FRET1_path)},原始fret2图片数：{len(FRET2_path)},' )
    return pred_FRET1, image_FRET1, image_FRET2


# In[29]:


"""
Big-Loader
"""

def load_exp(mode,root_dir,seg_folder_name,extend1,extend2,getall=False,watershed=False):

    print(">>>> Loding the predition, f1 image, f2 image <<<<")
    pred_FRET1, image_FRET1, image_FRET2 = dataloader_pred_fret1_fret2(root_dir, seg_folder_name, extend1,extend2)
    print(f">>>> start to compute the info of mode: {mode} all the images in exp： cell <<<<")
    if watershed:
        labeled_preds, preds_info=get_labeled_preds_and_info(pred_FRET1,mode)
    else:
        labeled_preds, preds_info=get_labeled_preds_and_info_no_watershed(pred_FRET1,mode)
    print("labeled Image:", len(labeled_preds), "image info:",len(preds_info))
    labeled_pred1 = labeled_preds[0] # 标注模式
    pred1_info = get_labeled_info(labeled_pred1)

    exp=labeled_preds,preds_info,image_FRET1,image_FRET2

    print(" cell numbers in first image:", len(pred1_info),'\n',"cell info in first image:",np.max(labeled_pred1),'\n')
    plt.figure(figsize=(24,24))
    plt.subplot(2,2,1)
    plt.imshow(labeled_pred1)
    plt.subplot(2,2,2)
    plt.imshow(image_FRET1[0])
    if getall:
        return exp,labeled_pred1,pred1_info
    else:
        return exp


# In[194]:


"""
exp1

"""


root_dir_1 = r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_1_F_cube_4_1\Pos0"
seg_folder_name_1 = 'Unet++_of_434_Norm_withmodel-0.707_orig0413_pth'
extend1_1 = '_438-542-4_6_000'
extend2_1 = '_438-483-4_000'

"""
exp2
"""
root_dir_2 = r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_2_F_cube_4_1\Pos0"             
seg_folder_name_2 = 'Unet++_of_434_Norm_withmodel-0.707_orig0413_pth'
extend1_2 = '_438-542-4_6_000'
extend2_2 = '_438-483-4_000'

"""
exp3
"""
root_dir_3 = r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_3_F_cube_4_1\Pos0"
seg_folder_name_3 = 'Unet++_of_434_Norm_withmodel-0.707_orig0413_pth'
extend1_3 = '_438-542-4_6_000'
extend2_3 = '_438-483-4_000'

"""
exp4
"""
root_dir_4 = r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_4_F_cube_4_1\Pos0"
seg_folder_name_4 = 'Unet++_of_434_Norm_withmodel-0.707_orig0413_pth'
extend1_4 = '_438-542-4_6_000'
extend2_4 = '_438-483-4_000'
"""
exp5
"""
root_dir_5 =  r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_5_repeat_stress_F_cube_4_1\Pos0"
seg_folder_name_5 = 'Unet++_of_434_Norm_withmodel-0.707_orig0413_pth'
extend1_5 = '_438-542-4_6_000'
extend2_5 = '_438-483-4_000'

"""
exp6
"""
root_dir_6 =  r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_6_repeat_stress_F_cube_4_1\Pos0"
seg_folder_name_6 = 'Unet++_of_434_Norm_withmodel-0.707_orig0413_pth'
extend1_6 = '_438-542-4_6_000'
extend2_6 = '_438-483-4_000'


# In[195]:


# """
# plate 3
# """




# """
# exp1
# 地址 长的当做fret1 短的当做fret2
# always f2/f1
# """

# root_dir_1 = r"C:\Users\Mengxi\Box\Data\SingleCell\20220504_cellnumber_celltype\plate3_sample1_1\Pos0"
# seg_folder_name_1 = 'Unet++_of_434_justtophat_Norm_withmodel-0.707_orig0413_pth'
# extend1_1 = '_438-542-4_6_000'
# extend2_1 = '_438-483-4_000'

# """
# exp2
# """
# root_dir_2 = r"C:\Users\Mengxi\Box\Data\SingleCell\20220506_mix_cell_dtest_&_stress_test\repeat D_test_exp2_1\Pos0"
# seg_folder_name_2 = 'Unet++_of_434_justtophat_Norm_withmodel-0.707_orig0413_pth'
# extend1_2 = '_438-542-4_6_000'
# extend2_2 = '_438-483-4_000'

# """
# exp3
# """
# root_dir_3 = r"C:\Users\Mengxi\Box\Data\SingleCell\20220504_cellnumber_celltype\plate3_sample3_1\Pos0"
# seg_folder_name_3 = 'Unet++_of_434_justtophat_Norm_withmodel-0.707_orig0413_pth'
# extend1_3 = '_438-542-4_6_000'
# extend2_3 = '_438-483-4_000'

# """
# exp4
# """
# root_dir_4 = r"C:\Users\Mengxi\Box\Data\SingleCell\20220504_cellnumber_celltype\plate3_sample4_1\Pos0"
# seg_folder_name_4 = 'Unet++_of_434_justtophat_Norm_withmodel-0.707_orig0413_pth'
# extend1_4 = '_438-542-4_6_000'
# extend2_4 = '_438-483-4_000'

# """
# exp5
# """
# root_dir_4 = r"C:\Users\Mengxi\Box\Data\SingleCell\20220504_cellnumber_celltype\plate3_sample5_1\Pos0"
# seg_folder_name_4 = 'Unet++_of_434_justtophat_Norm_withmodel-0.707_orig0413.pth'
# extend1_4 = '_438-542-4_6_000'
# extend2_4 = '_438-483-4_000'

# """
# exp6
# """
# root_dir_4 = r"C:\Users\Mengxi\Box\Data\SingleCell\20220504_cellnumber_celltype\plate3_sample6_1\Pos0"
# seg_folder_name_4 = 'Unet++_of_434_justtophat_Norm_withmodel-0.707_orig0413.pth'
# extend1_4 = '_438-542-4_6_000'
# extend2_4 = '_438-483-4_000'


# In[162]:


"""
chose a mode
1 whole cell
2 neclei
3 cyto
"""

mode=1


# In[188]:


"""
每部分都有预览，一定要先判断样本是否合格，不合格就不要进行计算，会报错
Each part has a preview. 
must firstly judge whether the sample is qualified. 
If it is unqualified, do not calculate it, 
else an error will be reported
载入 exp1
"""
#timelist_1=[30,150,270,390,510]
timelist_1=[30,150,270]

exp1=load_exp(mode,root_dir_1,seg_folder_name_1,extend1_1,extend2_1,getall=False)


# In[189]:


"""
载入 exp2
"""
timelist_2=[30,150,270]
exp2=load_exp(mode,root_dir_2,seg_folder_name_2,extend1_2,extend2_2,getall=False)


# In[190]:


"""
载入 exp3
"""
timelist_3=[30,160,280]
exp3=load_exp(mode,root_dir_3,seg_folder_name_3,extend1_3,extend2_3,getall=False)


# In[191]:


"""
载入 exp4
"""
timelist_4=[30,160,280]
exp4=load_exp(mode,root_dir_4,seg_folder_name_4,extend1_4,extend2_4,getall=False)


# In[192]:


"""
载入 exp5
"""
timelist_5=[30,120,210,300,393,481,571]
exp5=load_exp(mode,root_dir_5,seg_folder_name_5,extend1_5,extend2_5,getall=False)


# In[195]:


"""
载入 exp6
"""
timelist_6=[30,150,240,330,450,571,690]
exp6=load_exp(mode,root_dir_6,seg_folder_name_6,extend1_6,extend2_6,getall=False)


# In[87]:



columns_agent_name_list1=["Diclofenac","Oligmycin & Diclofenac","Diclofenac","Diclofenac"]       

pd_full_slope_celldict_3,pd_full_fret_celldict_3=get_slopdict_fretdict(root_dir_3,celldict3,slopetimelist3,columns_agent_name_list3)                     


# In[219]:


"""
大部分的计算在这里完成
"""
celldict1,slopetimelist1,frets1,areas1,peri1,eccen1=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp1,timelist_1,celldict=True)
info_of_exp1=frets1,areas1,peri1,eccen1
celldict2,slopetimelist2,frets2,areas3,peri2,eccen2=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp2,timelist_2,celldict=True)                
info_of_exp2=frets2,areas3,peri2,eccen2
celldict3,slopetimelist3,frets3,areas3,peri3,eccen3=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp3,timelist_3,celldict=True)
info_of_exp3=frets3,areas3,peri3,eccen3
celldict4,slopetimelist4,frets4,areas4,peri4,eccen4=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp4,timelist_4,celldict=True)
info_of_exp4=frets4,areas4,peri4,eccen4


# In[221]:


celldict2,slopetimelist2,frets2,areas3,peri2,eccen2=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp2,timelist_2,celldict=True)                
info_of_exp2=frets2,areas3,peri2,eccen2


# In[222]:


celldict3,slopetimelist3,frets3,areas3,peri3,eccen3=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp3,timelist_3,celldict=True)
info_of_exp3=frets3,areas3,peri3,eccen3


# In[223]:


celldict4,slopetimelist4,frets4,areas4,peri4,eccen4=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp4,timelist_4,celldict=True)
info_of_exp4=frets4,areas4,peri4,eccen4


# In[197]:


celldict5,slopetimelist5,frets5,areas5,peri5,eccen5=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp5,timelist_5,celldict=True)
info_of_exp5=frets5,areas5,peri5,eccen5


# In[198]:


celldict6,slopetimelist6,frets6,areas6,peri6,eccen6=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp6,timelist_6,celldict=True)
info_of_exp6=frets6,areas6,peri6,eccen6


# In[310]:


"""
为画图准备slops数据 保存csv
注意修改保存名称和保存地址
"""

def save_slops_csv_png(slopetimelists,save_slops_dir,slops_file_name,barcode=False,figure=True):
    explist = []
    slopslist = []
    starttimelist = []
    print("Attention：")
    print(f"共有{len(slopetimelists)}个实验",)
    for expNo, singlecellsloptimelists in enumerate(slopetimelists):
        print(f"{expNo+1}实验有{len(singlecellsloptimelists)}个有效细胞，",)    
        
        #singlecellsloptimelists.all() =singlecellsloptimelists.all()[(np.abs(stats.zscore(singlecellsloptimelists.all())) < 1.64)]

        for celllabels,singletimeslopes in enumerate(singlecellsloptimelists):
            print(f"{celllabels}细胞有{len(singletimeslopes)}个有效时间点",)
            if len(singletimeslopes)!=0:
                for starttime,slop in enumerate(singletimeslopes):                    
                    exp = "Exp-" + str(expNo + 1)               
                    starttime = "Time Sequence-"+ str(starttime + 1)
                    explist.append(exp)
                    slopslist.append(slop)
                    starttimelist.append(starttime)

    explist = pd.DataFrame(explist)
    slopslist = pd.DataFrame(slopslist)
    starttimelist = pd.DataFrame(starttimelist)
    data = pd.concat([slopslist, explist, starttimelist], axis=1)
    print(data.shape)
    
    data.columns = (["slopes", "exp_batch_index", "time_serise"])
    print(data)
    if barcode:
        data.to_csv(os.path.join(save_slops_dir,"mode_"+str(mode)+str(barcode)+slops_file_name+".csv"),index=False,header=True)
        print(f"please find saved data in {save_slops_dir}")
        if figure:

            plt.figure(figsize=(12,6))
            sns.boxenplot(x="time_serise", y="slopes",hue="exp_batch_index",data=data)

            plt.xlabel("exp_batch_index")
            plt.ylabel("slopes")
            plt.savefig(os.path.join(save_slops_dir,"mode_"+str(mode)+str(barcode)+slops_file_name+"_time_serise.png"))
            sns.scatterplot(x="time_serise", y="slopes",hue="exp_batch_index",data=data)
            plt.xlabel("exp_batch_index")
            plt.ylabel("slopes")
            plt.savefig(os.path.join(save_slops_dir,"mode_"+str(mode)+str(barcode)+slops_file_name+"_time_serise_scatter.png"))

            plt.figure(figsize=(12,6))
            sns.violinplot(x="exp_batch_index", y="slopes",hue="time_serise",data=data)

            plt.xlabel("exp batch index")
            plt.ylabel("slopes")
            plt.savefig(os.path.join(save_slops_dir,"mode_"+str(mode)+str(barcode)+slops_file_name+"_exp.png"))
    else:
        data.to_csv(os.path.join(save_slops_dir,"mode_"+str(mode)+slops_file_name+".csv"),index=False,header=True)
        print(f"please find saved data in {save_slops_dir}")
        if figure:

            plt.figure(figsize=(12,6))
            sns.boxenplot(x="time_serise", y="slopes",hue="exp_batch_index",data=data)

            plt.xlabel("exp_batch_index")
            plt.ylabel("slopes")
            plt.savefig(os.path.join(save_slops_dir,"mode_"+str(mode)+slops_file_name+"_time_serise.png"))
            sns.scatterplot(x="time_serise", y="slopes",hue="time_serise",data=data)
            plt.xlabel("exp_batch_index")
            plt.ylabel("slopes")
            plt.savefig(os.path.join(save_slops_dir,"mode_"+str(mode)+slops_file_name+"_time_serise_scatter.png"))

            plt.figure(figsize=(12,6))
            sns.violinplot(x="exp_batch_index", y="slopes",hue="time_serise",data=data)

            plt.xlabel("exp batch index")
            plt.ylabel("slopes")
            plt.savefig(os.path.join(save_slops_dir,"mode_"+str(mode)+slops_file_name+"_exp.png"))
        

        
        
        

#=============================================================================================
slopetimelists = list([slopetimelist4])#,, slopetimelist2, slopetimelist3,slopetimelist4timelist4, timelist5,timelist6

save_slops_dir=root_dir_4# /watershed/'
slops_file_name='_slops_info_'
#==============================================================================================


save_slops_csv_png(slopetimelists,save_slops_dir,slops_file_name,figure=True)


# In[ ]:



labeled_preds_1,preds_info_1,image_FRET1_1,image_FRET2_1=exp1
labeled_preds_2,preds_info_2,image_FRET1_2,image_FRET2_2=exp2
labeled_preds_3,preds_info_3,image_FRET1_3,image_FRET2_3=exp3


# In[323]:


"""
input第n个实验的 necelltesti,area,perimeter,eccent
绘图图以及获取最终数据集
get plot and csv
需要定义
save_dir=root_dir_
file_name='nuclue_para_infos_df.csv'
保存的图片格式n-n.png
第一个数字表示mode 1：全 2细胞核 3细胞质，第二个数字0：fret，1：area 2:perimeter 3:eccentricity

"""
# fill_between: fill_between(x, y1, y2=0, where=None, interpolate=False, step=None, *, data=None, **kwargs)
# 利用plt模块的fill_between方法可以绘制出填充区域, where参数接受一个bool对象, 表示在哪些地方填充(bool为True的地方),\
# alpha是填充空间的透明度, x是水平轴上的点, y1是数据集竖直轴上的点, y2是要与y1在每一个水平轴点处计算差值然后填充这两部分的区域, y2的默认值是0,\
# interpolate只有在使用了where参数同时两条曲线交叉时才有效, 使用这个参数会把曲线交叉处也填充使得填充的更完整



def get_average_cell_info_of_exp(infos_of_exp,root_dir,file_name,barcode=False):
    
    info_list=[]
    image_name=['Fret.png','area.png','prerimeter.png','eccentrity.png']
    for lla,info in enumerate(list(infos_of_exp)):
        lllliseggs=[]
        ylist=[]
        for i in range(len(info)):
            lllliseggs.append(len(info[i]))       
            nsjeod=info[i]
            #nsjeod=nsjeod*(np.abs(stats.zscore(nsjeod)) < 3)
            nsjeod[nsjeod==0]=np.nan
            #nsjeod=smooth_list(nsjeod)
            ylist.append(nsjeod)
        print(np.max(lllliseggs),np.min(lllliseggs))       
        ylist1=pd.DataFrame(ylist)
        valid_cell_num=ylist1.shape[0]
        valid_frame_num=ylist1.shape[1]
        #ylist1=ylist1.fillna(method="bfill")

        print(np.array(ylist).shape)
        # 95%置信区间: 样本数>30 +- 2std  样本数小于30 +-2.447 std
        #ylist1=ylist1[(np.abs(stats.zscore(ylist1)))<2]
        imputer = KNNImputer(n_neighbors=10)
        ylist1=imputer.fit_transform(ylist1)
        ylist1=np.array(ylist1)[:,:(valid_frame_num-10)]
        y=np.mean(np.array(ylist1),axis=0)
        # SEM
        y2=y+(stats.sem(pd.DataFrame(ylist1),axis=0))
        y3=y-(stats.sem(pd.DataFrame(ylist1),axis=0))
    #   std
    #     y2=y+(stats.tstd(pd.DataFrame(ylist1),axis=0))
    #     y3=y-(stats.tstd(pd.DataFrame(ylist1),axis=0))
        x=range(np.array(y).shape[0])
        plt.figure(figsize=(18,6))
        plt.plot(x,y,x,y2,x,y3)
        plt.fill_between(x,y2,y3,color='blue',alpha=.15)
        if barcode:
            plt.savefig(os.path.join(root_dir,str(mode)+str(barcode)+image_name[lla]))
        else:
            plt.savefig(os.path.join(root_dir,str(mode)+image_name[lla]))
        info_list.append(y)

    info_list=pd.DataFrame(info_list).T
    info_list.columns=(["Fret","Area","Perimeter","eccentricity"])
    info_list.to_csv(os.path.join(root_dir,file_name),index=False,header=True)
    print(f"please find saved data in {root_dir}/{file_name}")



# In[205]:


file_name_5 = str(mode) + '_exp5_para_infos_df.csv'
get_average_cell_info_of_exp(info_of_exp5,root_dir_5,file_name_5)

file_name_6 = str(mode) + '_exp6_para_infos_df.csv'
get_average_cell_info_of_exp(info_of_exp6,root_dir_6,file_name_6)


# In[227]:






file_name_1 = str(mode) + '_exp1_para_infos_df.csv'
get_average_cell_info_of_exp(info_of_exp1,root_dir_1,file_name_1)

file_name_2 = str(mode) + '_exp2_para_infos_df.csv'
get_average_cell_info_of_exp(info_of_exp2,root_dir_2,file_name_2)

file_name_3 = str(mode) + '_exp3_para_infos_df.csv'
get_average_cell_info_of_exp(info_of_exp3,root_dir_3,file_name_3)

file_name_4 = str(mode) + '_exp4_para_infos_df.csv'
get_average_cell_info_of_exp(info_of_exp4,root_dir_4,file_name_4)


# In[77]:


"""
保存字典信息
save dict
"""

celldict1
pd1=pd.DataFrame.from_dict(celldict1)
pd1.to_csv(os.path.join(root_dir_1,"mode_"+str(mode)+"sigle_cell_infodict(Fret_Area_Peri_Ecc)"+".csv"),index=False,header=True)    
print(f"please find saved data in {root_dir_1}")

pd3=pd.DataFrame.from_dict(celldict3)
pd3.to_csv(os.path.join(root_dir_3,"mode_"+str(mode)+"sigle_cell_infodict(Fret_Area_Peri_Ecc)"+".csv"),index=False,header=True)    
print(f"please find saved data in {root_dir_3}")


# In[ ]:


"""
一个互动对话
"""
full_fret_celldict_1={}
for cell_Acc, one_cell_infos in celldict1.items():
    if len(one_cell_slopes_on_timelist)==5:
        full_slope_celldict_1[cell_Acc]=one_cell_slopes_on_timelist
goodkeys1=list(full_slope_celldict_1.keys())
message = ""
while message != 'quit':
    message = input("input cell num(input quit if you want to quit):")
show_single_cell_info(message, labeled_pred1, labeled_preds, pred1_info, preds_info, 80,
                      image_FRET1, image_FRET2, add_agent_frame, figure=True)


# In[78]:


print(slopetimelist1)
print(slopetimelist3)


# In[179]:


"""
制作fret和slope的字典

make fret/slope dict
"""


def get_slopdict_fretdict(root_dir,celldict,slopetimelist,columns_agent_name_list):
    keys=list(celldict.keys())
    how_many_time_seq=len(columns_agent_name_list)
    print("细胞字典的所有有效细胞序号",keys)
    slopdict=dict(zip(keys,slopetimelist))
    full_slope_celldict={}
    for cell_Acc, one_cell_slopes_on_timelist in slopdict.items():
        if len(one_cell_slopes_on_timelist)==how_many_time_seq:
            full_slope_celldict[cell_Acc]=one_cell_slopes_on_timelist
    goodkeys=list(full_slope_celldict.keys())
    print("how many good keys ?:",len(goodkeys))
    print("good keys are",goodkeys)

    pd_with_cell_label_full_slope_celldict=pd.DataFrame(full_slope_celldict).T
    pd_full_slope_celldict=pd.DataFrame(full_slope_celldict).T.reset_index(drop=True)
    pd_full_slope_celldict=np.array(pd_full_slope_celldict,dtype=float)
    pd_full_slope_celldict=pd.DataFrame(pd_full_slope_celldict)
    print(pd_full_slope_celldict.shape)
    if pd_full_slope_celldict.shape[1]==len(columns_agent_name_list):
        pd_full_slope_celldict.columns=columns_agent_name_list
    else:
        print("输入的数据列数和time sequence的名字长度不一致，请重新输入\n","The number of data columns entered is inconsistent with the name length of time sequence. Please re-enter!")  

    pd_full_slope_celldict.to_csv(os.path.join(root_dir,"mode_"+str(mode)+"_single_cell_slopes_dict(cell_with_full_slopes)"+".csv"),index=True,header=True) 
    print(pd_full_slope_celldict)
    

    full_fret_celldict={}
    for cell_key, one_cell_infos in celldict.items():
        if cell_key in goodkeys:
            full_fret_celldict[cell_key]= []
            for i in range(pd.DataFrame(celldict).shape[0]):
                full_fret_celldict[cell_key].append(one_cell_infos[i][0])

    pd_full_fret_celldict=pd.DataFrame(full_fret_celldict)
    print(pd_full_fret_celldict)    
    pd_full_fret_celldict.to_csv(os.path.join(root_dir,"mode_"+str(mode)+"_(full_slopes)_cell_fret_dict"+".csv"),index=True,header=True)

    print(f"please find the slope/fret dict file at:{root_dir}")
    
    return pd_full_slope_celldict,pd_full_fret_celldict


# In[160]:



columns_agent_name_list1=["Diclofenac","Oligmycin & Diclofenac","Oligmycin & Diclofenac","Oligmycin & Diclofenac"]       
pd_full_slope_celldict_1,pd_full_fret_celldict_1=get_slopdict_fretdict(root_dir_1,celldict1,slopetimelist1,columns_agent_name_list1) 
                     


# In[159]:


columns_agent_name_list2=["Diclofenac","Oligmycin & Diclofenac","Diclofenac","Diclofenac"]         
pd_full_slope_celldict_2,pd_full_fret_celldict_2=get_slopdict_fretdict(root_dir_2,celldict2,slopetimelist2,columns_agent_name_list2)
                                   


# In[157]:



columns_agent_name_list3=["Diclofenac","2DG & Diclofenac","2DG & Diclofenac","2DG & Diclofenac"]       

pd_full_slope_celldict_3,pd_full_fret_celldict_3=get_slopdict_fretdict(root_dir_3,celldict3,slopetimelist3,columns_agent_name_list3)                     


# In[156]:



columns_agent_name_list4=["Diclofenac","2DG & Diclofenac","Diclofenac","Diclofenac"]       

pd_full_slope_celldict_4,pd_full_fret_celldict_4=get_slopdict_fretdict(root_dir_4,celldict4,slopetimelist4,columns_agent_name_list4)                     


# In[30]:


"""
相关系数分析 for slope
"""

    
    
columns_agent_name_list1=["Diclofenac_1st_time","Diclofenac_2nd_time","Diclofenac_3rd_time","Diclofenac_4th_time","Diclofenac_5th_time"]       


keys1=list(celldict1.keys())

print("所有有效细胞序号在所有slopetimelist1",keys1)
slopdict1=dict(zip(keys1,slopetimelist1))
full_slope_celldict_1={}
for cell_Acc, one_cell_slopes_on_timelist in slopdict1.items():
    if len(one_cell_slopes_on_timelist)==5:
        full_slope_celldict_1[cell_Acc]=one_cell_slopes_on_timelist
goodkeys1=list(full_slope_celldict_1.keys())
print(len(goodkeys1))
print(goodkeys1)
print(full_slope_celldict_1)
print(full_slope_celldict_1[6])
pd_with_cell_label_full_slope_celldict_1=pd.DataFrame(full_slope_celldict_1).T
pd_full_slope_celldict_1=pd.DataFrame(full_slope_celldict_1).T.reset_index(drop=True)
pd_full_slope_celldict_1=np.array(pd_full_slope_celldict_1,dtype=float)
pd_full_slope_celldict_1=pd.DataFrame(pd_full_slope_celldict_1)
pd_full_slope_celldict_1.columns=["Diclofenac_1st_time","Diclofenac_2nd_time","Diclofenac_3rd_time","Diclofenac_4th_time","Diclofenac_5th_time"]   

pd_full_slope_celldict_1.to_csv(os.path.join(root_dir_1,"mode_"+str(mode)+"sigle_cell_slopes_dict1(71cell_with_full_5_slopes)"+".csv"),index=False,header=True) 
print(pd_full_slope_celldict_1)
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

          
          
def draw_corr(pd_full_slope_celldict,root_dir)
    sns.set_theme(style="white")
    # rs = np.random.RandomState(12)
    # dd = pd.DataFrame(data=rs.normal(size=(100, 26)),
    #                  columns=list(ascii_letters[26:]))

    d=pd_full_slope_celldict
    #d=d.iloc[0,:].to_frame()
    print(d)
    # Compute the correlation matrix
    corr1 = d.corr()
    corr1=pd.DataFrame(corr)
    """
    保存 注意修改地址
    """
    corr_root=root_dir

    corr1.to_csv(os.path.join(root_dir_1,"mode_"+str(mode)+"repeated_D_corr"+".csv"),index=True,header=True)
    # Generate a mask for the upper triangle
    #mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr,  cmap=cmap,  center=0,vmin=-1,vmax=1,
                square=True, linewidths=.5, )#mask=mask,cbar_kws={"shrink": .5}


# In[31]:





# In[85]:


"""
相关系数分析 for fret
"""


keys1=list(celldict1.keys())
print(len(keys1))
print("所有有效细胞序号在所有slopetimelist1",keys1)

full_fret_celldict_1={}
for cell_Acc, one_cell_infos in celldict1.items():
    if cell_Acc in goodkeys1:
        full_fret_celldict_1[cell_Acc]= []
        for i in range(620):
            full_fret_celldict_1[cell_Acc].append(one_cell_infos[i][0])

pd_full_fret_celldict_1=pd.DataFrame(full_fret_celldict_1)
print(pd_full_fret_celldict_1)
"""
保存 注意修改地址
"""
root=root_dir_1
pd_full_fret_celldict_1.to_csv(os.path.join(root,"mode_"+str(mode)+"_5_slopes_cell_fret_dict"+".csv"),index=True,header=True)
# print(len(goodkeys1))
# print(goodkeys1)
# print(full_slope_celldict_1)
# print(full_slope_celldict_1[6])
# pd_with_cell_label_full_slope_celldict_1=pd.DataFrame(full_slope_celldict_1).T
# pd_full_slope_celldict_1=pd.DataFrame(full_slope_celldict_1).T.reset_index(drop=True)
# pd_full_slope_celldict_1=np.array(pd_full_slope_celldict_1,dtype=float)
# pd_full_slope_celldict_1=pd.DataFrame(pd_full_slope_celldict_1)
# pd_full_slope_celldict_1.columns=["Diclofenac_1st_time","Diclofenac_2nd_time","Diclofenac_3rd_time","Diclofenac_4th_time","Diclofenac_5th_time"]   

# pd_full_slope_celldict_1.to_csv(os.path.join(root_dir_1,"mode_"+str(mode)+"sigle_cell_slopes_dict(71cell_with_full_5_slopes)"+".csv"),index=False,header=True) 
# print(pd_full_slope_celldict_1)
# from string import ascii_letters
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.set_theme(style="white")
# rs = np.random.RandomState(12)
# dd = pd.DataFrame(data=rs.normal(size=(100, 26)),
#                  columns=list(ascii_letters[26:]))

# d=pd_full_slope_celldict_1
# #d=d.iloc[0,:].to_frame()
# print(d)
# # Compute the correlation matrix
# corr = d.corr()

# # Generate a mask for the upper triangle
# #mask = np.triu(np.ones_like(corr, dtype=bool))

# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))

# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(230, 20, as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr,  cmap=cmap,  center=0,vmin=-1,vmax=1,
#             square=True, linewidths=.5, )#mask=mask,cbar_kws={"shrink": .5}


# In[49]:


print(pd.DataFrame(celldict1).shape[0])


# In[229]:


"""
get barcode mask need ori + pred

and rv mask
"""
def FillHole(pred_img,thresh):
    
    # 复制 im_in 图像
    ret, thresh_pred = cv.threshold(np.array(pred_img), thresh, 1, cv.THRESH_BINARY)
    im_in = np.array(thresh_pred,dtype='uint8')
    im_floodfill = im_in.copy()
    
    # Mask 用于 floodFill，官方要求长宽+2
    h, w = im_in.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # floodFill函数中的seedPoint对应像素必须是背景
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if(im_floodfill[i][j]==0):
                seedPoint=(i,j)
                isbreak = True
                break
        if(isbreak):
            break
    
    # 得到im_floodfill 非空洞值是固定的255，填充的数值是255-？=孔洞值
    #floodFill( 1.操作的图像, 2.掩模, 3.起始像素位置，4.要填充什么颜色, 5.填充颜色的低值， 6.填充颜色的高值 ,7.填充的方法)
    cv.floodFill(im_floodfill, mask,seedPoint, 255,1,0)

    # 得到im_floodfill的逆im_floodfill_inv ,包含所有空洞
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
#     plt.imshow(im_floodfill_inv)
    im_floodfill_inv_copy = im_floodfill_inv.copy()
    # 函数findContours获取轮廓
#     contours, hierarchy = cv.findContours(im_floodfill_inv_copy,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
#     SizeThreshold=30
#     for num in range(len(contours)):
#         if(cv.contourArea(contours[num])>=SizeThreshold):
#             cv.fillConvexPoly(im_floodfill_inv, contours[num], 0)

    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = np.array(im_in) | im_floodfill_inv
#     plt.figure(figsize=(24,24))
#     print(np.unique(im_out))
    
    ret, thresh_pred = cv.threshold(np.array(im_out), 254, 1, cv.THRESH_BINARY)
#     print(np.unique(thresh_pred))
#     plt.imshow(thresh_pred)
    return np.array(thresh_pred)


def get_barcode_mask(bc_pred_dir,bc_ori_dir,bar=1000):
    image=Image.open(bc_pred_dir)
    ret, thresh_barcode = cv.threshold(np.array(image), 1, 1, cv.THRESH_BINARY)
    kernel = np.ones((1,1),np.uint8)
    thresh_barcode = cv.dilate(thresh_barcode,kernel,iterations = 1)
    erosion = FillHole(thresh_barcode,0)
    #contours,hierarchy = cv.findContours(erosion, 1, 2)
    rett,barcodemask= cv.threshold(tifffile.imread(bc_ori_dir), bar, 1, cv.THRESH_BINARY)
    #barcodemask = cv.erode(barcodemask,np.ones((1,1),np.uint8),iterations = 1)
    plt.figure(figsize=(24,24))
    plt.subplot(2,2,1)
    plt.imshow(image)
    plt.subplot(2,2,2)
    plt.imshow(erosion)
    plt.subplot(2,2,3)
    plt.imshow(barcodemask)
    plt.subplot(2,2,4)
    plt.imshow(barcodemask*erosion)
    return barcodemask*erosion

def get_rv_mask(barcodemask):
    kernel = np.ones((1,1),np.uint8)
    rv_barcodemask = cv.dilate(barcodemask,kernel,iterations = 1)
    return 1-rv_barcodemask


# In[293]:


"""
barcode dir position
get barcode mask
"""
bc_pred_dir_1_r=r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_1_R_cube_5_1\Pos0\Unet++_of_434_Norm_withmodel-0.707_orig0413_pth\img_000000000_TRITC-Cube5_000.png"
bc_pred_dir_1_b=r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_1_B_cube_1_1\Pos0\Unet++_of_434_Norm_withmodel-0.707_orig0413_pth\img_000000000_DAPI-1 GFP-2 TRITC-3_000.png"

bc_pred_dir_2_r=r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_2_R_cube_5_1\Pos0\Unet++_of_434_Norm_withmodel-0.707_orig0413_pth\img_000000000_TRITC-Cube5_000.png"                                                  
bc_pred_dir_2_b=r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_2_B_cube_1_1\Pos0\Unet++_of_434_Norm_withmodel-0.707_orig0413_pth\img_000000000_DAPI-1 GFP-2 TRITC-3_000.png"

bc_pred_dir_3_r=r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_3_R_cube_5_1\Pos0\Unet++_of_434_Norm_withmodel-0.707_orig0413_pth\img_000000000_TRITC-Cube5_000.png"
bc_pred_dir_3_b=r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_3_B_cube_1_1\Pos0\Unet++_of_434_Norm_withmodel-0.707_orig0413_pth\img_000000000_DAPI-1 GFP-2 TRITC-3_000.png"

bc_pred_dir_4_r=r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_4_R_cube_5_1\Pos0\Unet++_of_434_Norm_withmodel-0.707_orig0413_pth\img_000000000_TRITC-Cube5_000.png"
bc_pred_dir_4_b=r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_4_B_cube_1_1\Pos0\Unet++_of_434_Norm_withmodel-0.707_orig0413_pth\img_000000000_DAPI-1 GFP-2 TRITC-3_000.png"

# bc_pred_dir_5=r"C:\Users\Mengxi\Box\Data\SingleCell\20220401_siMETTL3_DF\Area5_dsRed_1\Pos0\Unet++_of_434_Add_Neo_Norm_training_20220405\img_000000000_TRITC-Cube5_000.png"
# bc_pred_dir_6=r"C:\Users\Mengxi\Box\Data\SingleCell\20220401_siMETTL3_DF\Area6_dsRed_1\Pos0\Unet++_of_434_Add_Neo_Norm_training_20220405\img_000000000_TRITC-Cube5_000.png"

bc_ori_dir_1_r=r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_1_R_cube_5_1\Pos0\img_000000000_TRITC-Cube5_000.tif"
bc_ori_dir_1_b=r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_1_B_cube_1_1\Pos0\img_000000000_DAPI-1 GFP-2 TRITC-3_000.tif"

bc_ori_dir_2_r=r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_2_R_cube_5_1\Pos0\img_000000000_TRITC-Cube5_000.tif"
bc_ori_dir_2_b=r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_2_B_cube_1_1\Pos0\img_000000000_DAPI-1 GFP-2 TRITC-3_000.tif"

bc_ori_dir_3_r=r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_3_R_cube_5_1\Pos0\img_000000000_TRITC-Cube5_000.tif"
bc_ori_dir_3_b=r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_3_B_cube_1_1\Pos0\img_000000000_DAPI-1 GFP-2 TRITC-3_000.tif"


bc_ori_dir_4_r=r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_4_R_cube_5_1\Pos0\img_000000000_TRITC-Cube5_000.tif"
bc_ori_dir_4_b=r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_4_B_cube_1_1\Pos0\img_000000000_DAPI-1 GFP-2 TRITC-3_000.tif"

# bc_ori_dir_5=r"C:\Users\Mengxi\Box\Data\SingleCell\20220401_siMETTL3_DF\Area5_dsRed_1\Pos0\img_000000000_TRITC-Cube5_000.tif"
# bc_ori_dir_6=r"C:\Users\Mengxi\Box\Data\SingleCell\20220401_siMETTL3_DF\Area6_dsRed_1\Pos0\img_000000000_TRITC-Cube5_000.tif"

barcodemask1_r=get_barcode_mask(bc_pred_dir_1_r, bc_ori_dir_1_r,5000)
barcodemask1_b=get_barcode_mask(bc_pred_dir_1_b, bc_ori_dir_1_b,1000)

barcodemask2_r=get_barcode_mask(bc_pred_dir_2_r, bc_ori_dir_2_r,1000)
barcodemask2_b=get_barcode_mask(bc_pred_dir_2_b, bc_ori_dir_2_b,1000)

barcodemask3_r=get_barcode_mask(bc_pred_dir_3_r, bc_ori_dir_3_r,5000)
barcodemask3_b=get_barcode_mask(bc_pred_dir_3_b, bc_ori_dir_3_b,2000)

barcodemask4_r=get_barcode_mask(bc_pred_dir_4_r, bc_ori_dir_4_r,5000)
barcodemask4_b=get_barcode_mask(bc_pred_dir_4_b, bc_ori_dir_4_b,1000)


# In[ ]:





# In[300]:


"""
加入分水岭+mask 的细胞label
watershed added label
mask=0 no maks

mask=1 Redbarcoded cell

mask=2 None R cell

"""

def make_3ch_image(img):
    
    wozhenshixiande=np.zeros((img.shape[0],img.shape[1],3))
    wozhenshixiande[:,:,0]=img
    wozhenshixiande[:,:,1]=img
    wozhenshixiande[:,:,2]=img
    return np.array(wozhenshixiande,dtype='uint8')
    

def get_labeled_pred_barcode_watershed(pred_img,barcodemask1,barcodemask2,thresh=1,getallcell=False):

    img=cv.imread(pred_img)
    pred_img=Image.open(pred_img)
        
    gray = np.array(pred_img)# cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, gray = cv.threshold(gray, thresh, 1, cv.THRESH_BINARY)
#     plt.figure(figsize=(42,42))
#     plt.imshow(gray)
    ret, thresh = cv.threshold(gray,0,1,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    thresh=1-thresh
#     plt.figure(figsize=(3,3))
#     plt.imshow(thresh)
    # 噪声去除
    kernel = np.ones((5,5),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    opening=np.array(opening,dtype="uint8")
    # 确定背景区域
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    # 寻找前景区域
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.1*dist_transform.max(),1,0)
    # 找到未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
#     plt.figure(figsize=(8,8))
#     plt.imshow(unknown)
    # 类别标记
    ret, markers = cv.connectedComponents(sure_fg)
#     plt.figure(figsize=(7,7))
#     plt.imshow(markers+1)
    # 为所有的标记加1，保证背景是0而不是1
    markers = markers+1
    # 现在让所有的未知区域为0
    markers[unknown==1] = 0
    markers = cv.watershed(img,markers) 
    gray[markers == -1] = [0]
    
    
    size=np.array(pred_img).shape
    ret, thresh_pred = cv.threshold(np.array(gray), 0, 1, cv.THRESH_BINARY)
    #形态学调整
#     kernel = np.ones((4,4),np.uint8)
#     closing = cv.morphologyEx(thresh_pred, cv.MORPH_CLOSE, kernel)
#     kernel2 = np.ones((10,10),np.uint8)
#     thresh_pred = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel2)
#    plt.imshow(thresh_pred)
    labels_objects = measure.label(thresh_pred, background=0, connectivity=1)

    properties = measure.regionprops(labels_objects)
    object_areas = []
    object_labels = []
    object_coords = []
    object_cent = []
    
    
    
    for prop in properties:
        object_areas.append(prop.area)
        object_labels.append(prop.label)
        object_coords.append(prop.coords)
        object_cent.append(prop.centroid)
    props=list(zip(object_labels,object_areas,object_coords,object_cent))
    #limitmax=np.mean(object_areas) + (np.std(object_areas)*3)      #3
    limitmin=np.mean(object_areas) * 0.5 #(np.std(object_areas)*0.5)    #0.1
    
    #print(np.array(pred_img)[int(object_cent[0][0]),int(object_cent[0][1])])

    
    
    valid_coords1=[]
    valid_coordsNot1=[]
    valid_coords2=[]
    valid_coordsNot2=[]
    valid_coordsAll=[]
    
    for label,prop in enumerate(props):
        labels=prop[0]
        area=prop[1]
        coords=prop[2]
        cent=prop[3]        
        if  area>limitmin: #area<limitmax and
            if getallcell==False:
                if (np.array(pred_img)*barcodemask1)[int(cent[0]),int(cent[1])]==0:
                    valid_coordsNot1.append(coords)
                if (np.array(pred_img)*barcodemask1)[int(cent[0]),int(cent[1])]!=0:
                    valid_coords1.append(coords)
                if (np.array(pred_img)*barcodemask2)[int(cent[0]),int(cent[1])]==0:
                    valid_coordsNot2.append(coords)
                if (np.array(pred_img)*barcodemask2)[int(cent[0]),int(cent[1])]!=0:
                    valid_coords2.append(coords)
            else:
                valid_coordsAll.append(coords)
    imgnew1=np.zeros((size))
    for K in valid_coords1:
        for k in K:
            imgnew1[k[0],k[1]]=1
    valid_labels_objects1=measure.label(imgnew1,background=0,connectivity=1)
    imgnew2=np.zeros((size))
    for M in valid_coordsNot1:
        for m in M:
            imgnew2[m[0],m[1]]=1
    valid_labels_objectsNot1=measure.label(imgnew2,background=0,connectivity=1)
    imgnew3=np.zeros((size))
    for O in valid_coordsAll:
        for o in O:
            imgnew3[o[0],o[1]]=1
    valid_labels_objectsALL=measure.label(imgnew3,background=0,connectivity=1)
    imgnew4=np.zeros((size))
    for Q in valid_coords2:
        for q in Q :
            imgnew4[q[0],q[1]]=1
    valid_labels_objects2=measure.label(imgnew4,background=0,connectivity=1)
    imgnew5=np.zeros((size))
    for P in valid_coordsNot2:
        for p in P:
            imgnew5[p[0],p[1]]=1
    valid_labels_objectsNot2=measure.label(imgnew5,background=0,connectivity=1)  
    
    if getallcell:
        return valid_labels_objectsALL
    else:
        return valid_labels_objects1,valid_labels_objectsNot1,valid_labels_objects2,valid_labels_objectsNot2


# In[302]:



pred_img=r"C:\Users\Mengxi\Box\Data\SingleCell\20220519_bar_code_r&b_repeated_stress_test_telo\sample_3_F_cube_4_1\Pos0\Unet++_of_434_Norm_withmodel-0.707_orig0413_pth\img_000000000_438-542-4_6_000.png"
barcodemask1=barcodemask3_r
barcodemask2=barcodemask3_b

code1_R,code1_NR,code1_B,code1_NB=get_labeled_pred_barcode_watershed(pred_img,barcodemask1,barcodemask2,thresh=1,getallcell=False)

plt.imshow(code1_B)


# In[ ]:





# In[296]:


"""
加入watershed barcode
获得纯细胞质区域（去除细胞核）
"""

def get_labeled_pred_only_cyto_watershed(pred_img,barcodemask1,barcodemask2,getallcell=False): 
    img=cv.imread(pred_img)
    pred_img=Image.open(pred_img)
    
    size=np.array(pred_img).shape
    thresh_nuclei=2
    thresh_cyto=1
    ret, thresh_pred_neclei = cv.threshold(np.array(pred_img), thresh_nuclei, 1, cv.THRESH_BINARY)
    
    ret, thresh_pred_cyto = cv.threshold(np.array(pred_img), thresh_cyto, 1, cv.THRESH_BINARY)
    
    gray=thresh_pred_cyto
    #形态学调整
#     kernel = np.ones((4,4),np.uint8)
#     closing = cv.morphologyEx(thresh_pred, cv.MORPH_CLOSE, kernel)
#     kernel2 = np.ones((10,10),np.uint8)
#     thresh_pred = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel2)
    ret, thresh = cv.threshold(gray,0,1,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    thresh=1-thresh
#     plt.figure(figsize=(3,3))
#     plt.imshow(thresh)
    # 噪声去除
    kernel = np.ones((5,5),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    # 确定背景区域
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    # 寻找前景区域
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.1*dist_transform.max(),1,0)
    # 找到未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
#     plt.figure(figsize=(8,8))
#     plt.imshow(unknown)
    # 类别标记
    ret, markers = cv.connectedComponents(sure_fg)
#     plt.figure(figsize=(7,7))
#     plt.imshow(markers+1)
    # 为所有的标记加1，保证背景是0而不是1
    markers = markers+1
    # 现在让所有的未知区域为0
    markers[unknown==1] = 0
    markers = cv.watershed(img,markers) 
    thresh_pred_cyto[markers == -1] = [0]
    
    thresh_pred=thresh_pred_cyto-thresh_pred_neclei
    #形态学调整
#     kernel = np.ones((4,4),np.uint8)
#     closing = cv.morphologyEx(thresh_pred, cv.MORPH_CLOSE, kernel)
#     kernel2 = np.ones((10,10),np.uint8)
#     thresh_pred = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel2)
    
    
    labels_objects = measure.label(thresh_pred, background=0, connectivity=1)

    properties = measure.regionprops(labels_objects)
    object_areas = []
    object_labels = []
    object_coords = []
    object_cent = []
   
    for prop in properties:
        object_areas.append(prop.area)
        object_labels.append(prop.label)
        object_coords.append(prop.coords)
        object_cent.append(prop.centroid)
    props=list(zip(object_labels,object_areas,object_coords,object_cent))
    #limitmax=np.mean(object_areas) + (np.std(object_areas)*3)      #3
    limitmin=np.mean(object_areas) * 0.5 #(np.std(object_areas)*0.5)    #0.1
    
    #print(np.array(pred_img)[int(object_cent[0][0]),int(object_cent[0][1])])
   
    valid_coords1=[]
    valid_coordsNot1=[]
    valid_coords2=[]
    valid_coordsNot2=[]
    valid_coordsAll=[]
    
    for label,prop in enumerate(props):
        labels=prop[0]
        area=prop[1]
        coords=prop[2]
        cent=prop[3]        
        if  area>limitmin: #area<limitmax and
            if getallcell==False:
                if (np.array(pred_img)*barcodemask1)[int(cent[0]),int(cent[1])]==0:
                    valid_coordsNot1.append(coords)
                elif (np.array(pred_img)*barcodemask1)[int(cent[0]),int(cent[1])]!=0:
                    valid_coords1.append(coords)
                elif (np.array(pred_img)*barcodemask2)[int(cent[0]),int(cent[1])]==0:
                    valid_coordsNot2.append(coords)
                elif (np.array(pred_img)*barcodemask2)[int(cent[0]),int(cent[1])]!=0:
                    valid_coords2.append(coords)
            else:
                valid_coordsAll.append(coords)
    imgnew1=np.zeros((size))
    for K in valid_coords1:
        for k in K:
            imgnew1[k[0],k[1]]=1
    valid_labels_objects1=measure.label(imgnew1,background=0,connectivity=1)
    imgnew2=np.zeros((size))
    for M in valid_coordsNot1:
        for m in M:
            imgnew2[m[0],m[1]]=1
    valid_labels_objectsNot1=measure.label(imgnew2,background=0,connectivity=1)
    imgnew3=np.zeros((size))
    for O in valid_coordsAll:
        for o in O:
            imgnew3[o[0],o[1]]=1
    valid_labels_objectsALL=measure.label(imgnew3,background=0,connectivity=1)
    imgnew4=np.zeros((size))
    for Q in valid_coords2:
        for q in Q :
            imgnew4[q[0],q[1]]=1
    valid_labels_objects2=measure.label(imgnew4,background=0,connectivity=1)
    imgnew5=np.zeros((size))
    for P in valid_coordsNot2:
        for p in P:
            imgnew5[p[0],p[1]]=1
    valid_labels_objectsNot2=measure.label(imgnew5,background=0,connectivity=1)  
    
    if getallcell:
        return valid_labels_objectsALL
    else:
        return valid_labels_objects1,valid_labels_objectsNot1,valid_labels_objects2,valid_labels_objectsNot2            


# In[325]:


def get_labeled_preds_and_info_barcode(pred_list,barcodemask1,barcodemask2,mode,getNonecode=False):

    labeled_preds_R = []
    labeled_preds_B = []
    preds_info_R = []
    preds_info_B = []
    labeled_preds_NR = []
    labeled_preds_NB = []
    preds_info_NR = []
    preds_info_NB = []    
    
    
    if mode==3:
        for pred in tqdm(pred_list):
            labeled_pred_cell_R,labeled_pred_cell_NR,labeled_pred_cell_B,labeled_pred_cell_NB = get_labeled_pred_only_cyto_watershed(pred,barcodemask1,barcodemask2,thresh=2)

            labeled_preds_R.append(labeled_pred_cell_R)
            labeled_preds_B.append(labeled_pred_cell_B)
            preds_info_R.append(get_labeled_info(labeled_pred_cell_R))
            preds_info_B.append(get_labeled_info(labeled_pred_cell_B))
            if getNonecode:
                labeled_preds_NR.append(labeled_pred_cell_NR)
                labeled_preds_NB.append(labeled_pred_cell_NB)
                preds_info_NR.append(get_labeled_info(labeled_pred_cell_NR))
                preds_info_NB.append(get_labeled_info(labeled_pred_cell_NB))    
    else:
        for pred in tqdm(pred_list):
            labeled_pred_cell_R,labeled_pred_cell_NR,labeled_pred_cell_B,labeled_pred_cell_NB = get_labeled_pred_barcode_watershed(pred, barcodemask1,barcodemask2,thresh=mode)

            labeled_preds_R.append(labeled_pred_cell_R)
            labeled_preds_B.append(labeled_pred_cell_B)
            preds_info_R.append(get_labeled_info(labeled_pred_cell_R))
            preds_info_B.append(get_labeled_info(labeled_pred_cell_B))
            if getNonecode:
                labeled_preds_NR.append(labeled_pred_cell_NR)
                labeled_preds_NB.append(labeled_pred_cell_NB)
                preds_info_NR.append(get_labeled_info(labeled_pred_cell_NR))
                preds_info_NB.append(get_labeled_info(labeled_pred_cell_NB))
    if getNonecode:
        return labeled_preds_R,labeled_preds_B,preds_info_R,preds_info_B,labeled_preds_NR,labeled_preds_NB,preds_info_NR,preds_info_NB 
    else:
        return labeled_preds_R,labeled_preds_B,preds_info_R,preds_info_B



def load_barcode_exp_watershed(mode,root_dir,seg_folder_name,extend1,extend2,barcode1,barcode2,getNonecode=False,getall=False):

    print(">>>> Loding the predition, f1 image, f2 image <<<<")
    pred_FRET1, image_FRET1, image_FRET2 = dataloader_pred_fret1_fret2(root_dir, seg_folder_name, extend1,extend2)
    print(f">>>> start to compute the info of mode: {mode} all the images in exp： cell <<<<")
    if getNonecode:
        labeled_preds_R,labeled_preds_B,preds_info_R,preds_info_B,labeled_preds_NR,labeled_preds_NB,preds_info_NR,preds_info_NB=get_labeled_preds_and_info_barcode(pred_FRET1,barcode1,barcode2,mode,1)
    else:
        labeled_preds_R,labeled_preds_B,preds_info_R,preds_info_B=get_labeled_preds_and_info_barcode(pred_FRET1,barcode1,barcode2,mode)
    
    print("labeled Image bar-code 1:", len(labeled_preds_R), "image info bar-code 1:",len(preds_info_R),"labeled Image bar-code 2:", len(labeled_preds_B), "image info bar-code2:",len(preds_info_B))
    labeled_pred1_R = labeled_preds_R[0] # 标注模式
    pred1_info_R = get_labeled_info(labeled_pred1_R)
    labeled_pred1_B = labeled_preds_B[0] # 标注模式
    pred1_info_B = get_labeled_info(labeled_pred1_B)
    if getNonecode:
        print("labeled Image bar-code None 1:", len(labeled_preds_NR), "image info bar-code None 1:",len(preds_info_NR),"labeled Image bar-code None 2:", len(labeled_preds_NB), "image info bar-code None 2:",len(preds_info_NB))
        labeled_pred1_NR = labeled_preds_NR[0] # 标注模式
        pred1_info_NR = get_labeled_info(labeled_pred1_NR)
        labeled_pred1_NB = labeled_preds_NB[0] # 标注模式
        pred1_info_NB = get_labeled_info(labeled_pred1_NB)
    
    if getNonecode:
        exp_R=labeled_preds_R,preds_info_R,image_FRET1,image_FRET2
        exp_NR=labeled_preds_NR,preds_info_NR,image_FRET1,image_FRET2
        exp_B=labeled_preds_B,preds_info_B,image_FRET1,image_FRET2
        exp_NB=labeled_preds_NB,preds_info_NB,image_FRET1,image_FRET2        
    else:
        exp_R=labeled_preds_R,preds_info_R,image_FRET1,image_FRET2
        exp_B=labeled_preds_B,preds_info_B,image_FRET1,image_FRET2

    if getNonecode:
        print("None bar-code 1 cell numbers in first image:", len(pred1_info_NR),'\n',"None bar-code 1 cell info in first image:",np.max(labeled_pred1_NR),'\n')
        plt.figure(figsize=(24,24))
        plt.subplot(2,2,1)
        plt.imshow(labeled_pred1_NR)
        plt.subplot(2,2,2)
        plt.imshow(image_FRET1[0])
        print("None bar-code 2 cell numbers in first image:", len(pred1_info_NB),'\n',"bar-code 2 cell info in first image:",np.max(labeled_pred1_NB),'\n')
        plt.subplot(2,2,3)
        plt.imshow(labeled_pred1_NB)
        plt.subplot(2,2,4)
        plt.imshow(image_FRET1[0]) 
    
    else:
        print("bar-code 1 cell numbers in first image:", len(pred1_info_R),'\n',"bar-code 1 cell info in first image:",np.max(labeled_pred1_R),'\n')
        plt.figure(figsize=(24,24))
        plt.subplot(2,2,1)
        plt.imshow(labeled_pred1_R)
        plt.subplot(2,2,2)
        plt.imshow(image_FRET1[0])
        print("bar-code 2 cell numbers in first image:", len(pred1_info_B),'\n',"bar-code 2 cell info in first image:",np.max(labeled_pred1_B),'\n')
        plt.subplot(2,2,3)
        plt.imshow(labeled_pred1_B)
        plt.subplot(2,2,4)
        plt.imshow(image_FRET1[0]) 
        
    if getNonecode:
        return exp_R,exp_NR,exp_B,expNB
    if getall:
        return exp_R,labeled_pred1_R,pred1_info_R,exp_B,labeled_pred1_B,pred1_info_B
    else:
        return exp_R,exp_B


# In[281]:



"""
载入 exp1 barcode
"""

exp1_R,exp1_B=load_barcode_exp_watershed(mode,root_dir_1,seg_folder_name_1,extend1_1,extend2_1,barcodemask1_r,barcodemask1_b,getall=False)


# In[ ]:


exp1_R,exp1_NR,exp1_B,exp1_NB=load_barcode_exp_watershed(mode,root_dir_1,seg_folder_name_1,extend1_1,extend2_1,barcodemask1_r,barcodemask1_b,getNonecode=True,getall=False)


# In[282]:



"""
载入 exp2 barcode
"""
exp2_R,exp2_B=load_barcode_exp_watershed(mode,root_dir_2,seg_folder_name_2,extend1_2,extend2_2,barcodemask2_r,barcodemask2_b,getall=False)


# In[ ]:


exp2_R,exp2_NR,exp2_B,exp2_NB=load_barcode_exp_watershed(mode,root_dir_2,seg_folder_name_2,extend1_2,extend2_2,barcodemask2_r,barcodemask2_b,getNonecode=True,getall=False)


# In[87]:



columns_agent_name_list1=["Diclofenac","Oligmycin & Diclofenac","Diclofenac","Diclofenac"]       

pd_full_slope_celldict_3,pd_full_fret_celldict_3=get_slopdict_fretdict(root_dir_3,celldict3,slopetimelist3,columns_agent_name_list3)                     


# In[301]:



"""
载入 exp3 barcode
"""
exp3_R,exp3_B=load_barcode_exp_watershed(mode,root_dir_3,seg_folder_name_3,extend1_3,extend2_3,barcodemask3_r,barcodemask3_b,getall=False)


# In[ ]:


exp3_R,exp3_NR,exp3_B,exp3_NB=load_barcode_exp_watershed(mode,root_dir_3,seg_folder_name_3,extend1_3,extend2_3,barcodemask3_r,barcodemask3_b,getNonecode=True,getall=False)
exp4_R,exp4_NR,exp4_B,exp4_NB=load_barcode_exp_watershed(mode,root_dir_4,seg_folder_name_4,extend1_4,extend2_4,barcodemask4_r,barcodemask4_b,getNonecode=True,getall=False) 


# In[303]:



"""
载入 exp4 barcode
"""
exp4_R,exp4_B=load_barcode_exp_watershed(mode,root_dir_4,seg_folder_name_4,extend1_4,extend2_4,barcodemask4_r,barcodemask4_b,getall=False)


# In[284]:


"""
大部分的计算在这里完成 barcode
"""
celldict1_R,slopetimelist1_R,frets1_R,areas1_R,peri1_R,eccen1_R=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp1_R,timelist_1,celldict=True)
info_of_exp1_R=frets1_R,areas1_R,peri1_R,eccen1_R

# celldict1_B,slopetimelist1_B,frets1_B,areas1_B,peri1_B,eccen1_B=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp1_B,timelist_1,celldict=True)
# info_of_exp1_B=frets1_B,areas1_B,peri1_B,eccen1_B


# In[ ]:


celldict1_NR,slopetimelist1_NR,frets1_NR,areas1_NR,peri1_NR,eccen1_NR=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp1_NR,timelist_1,celldict=True)
info_of_exp1_NR=frets1_NR,areas1_NR,peri1_NR,eccen1_NR


# In[ ]:


celldict2_NR,slopetimelist2_NR,frets2_NR,areas2_NR,peri2_NR,eccen2_NR=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp2_NR,timelist_2,celldict=True)
info_of_exp2_NR=frets2_NR,areas2_NR,peri2_NR,eccen2_NR


# In[288]:


celldict2_R,slopetimelist2_R,frets2_R,areas2_R,peri2_R,eccen2_R=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp2_R,timelist_2,celldict=True)
info_of_exp2_R=frets2_R,areas2_R,peri2_R,eccen2_R

# celldict2_B,slopetimelist2_B,frets2_B,areas2_B,peri2_B,eccen2_B=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp2_B,timelist_2,celldict=True)
# info_of_exp2_B=frets2_B,areas2_B,peri2_B,eccen2_B




# In[ ]:





# In[291]:


celldict3_R,slopetimelist3_R,frets3_R,areas3_R,peri3_R,eccen3_R=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp3_R,timelist_3,celldict=True)
info_of_exp3_R=frets3_R,areas3_R,peri3_R,eccen3_R


# In[ ]:


celldict3_NR,slopetimelist3_NR,frets3_NR,areas3_NR,peri3_NR,eccen3_NR=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp3_NR,timelist_3,celldict=True)
info_of_exp3_NR=frets3_NR,areas3_NR,peri3_NR,eccen3_NR


# In[304]:


celldict3_B,slopetimelist3_B,frets3_B,areas3_B,peri3_B,eccen3_B=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp3_B,timelist_3,celldict=True)
info_of_exp3_B=B_tranform(frets3_B),areas3_B,peri3_B,eccen3_B


# In[318]:





# In[292]:




celldict4_R,slopetimelist4_R,frets4_R,areas4_R,peri4_R,eccen4_R=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp4_R,timelist_4,celldict=True)
info_of_exp4_R=frets4_R,areas4_R,peri4_R,eccen4_R


# In[ ]:


celldict4_NR,slopetimelist4_NR,frets4_NR,areas4_NR,peri4_NR,eccen4_NR=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp4_NR,timelist_4,celldict=True)
info_of_exp4_NR=frets4_NR,areas4_NR,peri4_NR,eccen4_NR


# In[305]:


celldict4_B,slopetimelist4_B,frets4_B,areas4_B,peri4_B,eccen4_B=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp4_B,timelist_4,celldict=True)
info_of_exp4_B=B_tranform(frets4_B),areas4_B,peri4_B,eccen4_B


# In[317]:


celldict4_NB,slopetimelist4_NB,frets4_NB,areas4_NB,peri4_NB,eccen4_NB=get_all_parlists_on_agentAddTimelist_of_One_exp(*exp4_NB,timelist_4,celldict=True)


# In[315]:


def B_tranform(slopetimelist):
    for i in slopetimelist:
        i=1/np.array(i)
    return slopetimelist

#=============================================================================================
slopetimelists = list([slopetimelist3_R,B_tranform(slopetimelist3_B)])#,, slopetimelist2, slopetimelist3,slopetimelist4timelist4, timelist5,timelist6

save_slops_dir=root_dir_3# /watershed/'
slops_file_name='_slops_info_RB'
#==============================================================================================



save_slops_csv_png(slopetimelists,save_slops_dir,slops_file_name,barcode="RB",figure=True)


# In[314]:





#=============================================================================================

slopetimelists = list([slopetimelist4_R,B_tranform(slopetimelist4_B)])#,, slopetimelist2, slopetimelist3,slopetimelist4timelist4, timelist5,timelist6

save_slops_dir=root_dir_4# /watershed/'
slops_file_name='_slops_info_RB'
#==============================================================================================


save_slops_csv_png(slopetimelists,save_slops_dir,slops_file_name,barcode="RB",figure=True)


# In[324]:


file_name_3R = str(mode) + '_exp3_R_para_infos_df.csv'
get_average_cell_info_of_exp(info_of_exp3_R,root_dir_3,file_name_3R,barcode="R")
file_name_3B = str(mode) + '_exp3_B_para_infos_df.csv'
get_average_cell_info_of_exp(info_of_exp3_B,root_dir_3,file_name_3B,barcode="B")

file_name_4R = str(mode) + '_exp4_R_para_infos_df.csv'
get_average_cell_info_of_exp(info_of_exp4_R,root_dir_4,file_name_4R,barcode="R")
file_name_4B = str(mode) + '_exp3_B_para_infos_df.csv'
get_average_cell_info_of_exp(info_of_exp4_B,root_dir_4,file_name_4B,barcode="B")

