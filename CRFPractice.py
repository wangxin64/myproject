from _collections import defaultdict
import random
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from time import *
#coding: utf-8


beginTime=time()
#SG的L2正则化系数和学习率
l2_coeff=1
rate=10
n=1000  #用户数量
m=1000  #物品数量
alpha=np.random.randn()+1  #调节随机值，为信任值添加随机因子
trust_alpha=np.random.random()  #信任阈值


#factor graph Constant
DEBUG_DEFAULT = True
E_STOP = False
LBP_MAX_ITERS = 300
unicode='UTF-8'


#构建信任矩阵,用数字代表用户，如1号用户，2号用户
similarMatrix=np.ones((m,m)) #物品相似度矩阵

itemScore=[]
similarMatrixOne=[]       #相似系数中间列表
similarCorrcoefMatrix=np.ones((n,n))  #用户i与用户j的相似系数
i=0
k=0
tag=0
k = 0
UserFriendSort={} #记录普通用户队
UsermimicSort={} #记录亲密用户队
#计算物品相似性(依据用户-物品打分矩阵)，依据相似性排名寻找信任用户
#假设[i,j]为用户对物品的打分(评级)信息,假设每行有3个物品
itemNewScore=[]
mylabel=[]

#推荐列表,保存推荐给用户的物品编号
RecommendationList= {}
NoneZeroElementForLine={}
NoneZeroFriendToItem={}

#测试推荐的用户数量
testn=10

#将列表中的字符串转为整数
def convert_to_int(lists):
    return [int(el) if not isinstance(el,list) else convert_to_int(el) for el in lists]

#字符串转化为浮点数
def convert_to_float(lists):
    return [float(el) if not isinstance(el,list) else convert_to_int(el) for el in lists]

#统计文件行数，列数
def fileLineNum(filepath):
    count = 0
    for index, line in enumerate(open(filepath,'r')):
        count += 1
    return count

def fileColumnNum(filepath):
    ColumnNum=np.shape(filepath)[1]
    return ColumnNum

#计算余弦相似度的方法
def get_cos_similar(v1: list, v2: list):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)+1.01  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

#计算欧式距离
def calEuclideanDistance(vec1:list,vec2:list):
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return dist

#计算余弦相似度
def cos_sim(a:list,b:list):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = a.dot(b.T)/(a_norm * b_norm)
    return cos

#推荐指标计算函数



#从原始数据集中读取数据，构建用户评分矩阵,error_bad_lines=False, encoding='unicode_escape',
#,error_bad_lines=False,encoding='unicode_escape',header=None
preprocessfile=pd.read_csv("D:\matlab\Myratings.csv",error_bad_lines=False,encoding='unicode_escape',header=None
                           )

#df=pd.DataFrame(preprocessfile).corr(method='spearman')
dk=pd.DataFrame(preprocessfile)

dk=pd.DataFrame(preprocessfile)
ratingMatrix=np.random.random((n,n))
print(dk)
for line in range(n):
    mylabel.append(int(dk.iloc[line,1]))
    if int(dk.iloc[line,0])<n and int(dk.iloc[line,1])<m:
        ratingMatrix[int(dk.iloc[line,0])][int(dk.iloc[line,1])]=int(dk.iloc[line,2])

'''
ratingMatrix=np.random.random((n,80))
dk=dk.dropna()
print(dk)

for i in range(100):
    for j in range(80):
        if int(dk.iloc[i,j])==99:
            ratingMatrix[i][j]= 0
        if int(dk.iloc[i,j])<0:
            ratingMatrix[i][j]=abs(int(dk.iloc[i,j]))
'''

#UserFavouItem[i]:{用户i:[物品1，物品2，物品3,...],..} 物品按喜爱程度以此排序
UserFavouItem={}
for i in range(n):
    UiLikeList = sorted(enumerate(ratingMatrix[i]), key=lambda x: x[1],reverse=True)
    for ItemAndScore in range(len(UiLikeList)):#j:物品编号
        UserFavouItem.setdefault(i,[]).append(UiLikeList[ItemAndScore][0])


UserLine=[]
for i in range(n):
    PerLineElementSum=0
# for j in range(m)
    for j in range(80):
        PerLineElementSum+=ratingMatrix[i][j]
    if PerLineElementSum!=0:
        UserLine.append(i)

for s in UserLine:
    for k in UserLine:
        cos = cos_sim(ratingMatrix[s], ratingMatrix[k])
        similarCorrcoefMatrix[s][k] = 1 / (cos + 1)



#随机信任初值,非对称信任矩阵(来源于社会心理学,人与人之间的互动频率不充分)
trustMatrix=np.random.randint(1,5,size=(n,n))
FriendthresholdValue=0.012  #普通朋友cos相关系数阈值
mimicthresholdValue=0.3   #密友cos相关系数阈值


#用户亲密度排序，寻找cos最大的用户为最亲密用户，以此类推，并记录用户编号
#顺次计算每个用户的亲密度，并构建线性链CRF
for i in range(0,len(similarCorrcoefMatrix[1])):
    for k in range(0,fileColumnNum(similarCorrcoefMatrix)):
        trustcoefficient=similarCorrcoefMatrix[i][k]*ratingMatrix[i][k]/k
        #从文件中读取信任值,信任值大于阈值且评价了大于2个物品的用户作为py
        if trustcoefficient>FriendthresholdValue:
            FriendthresholdValue=similarCorrcoefMatrix[i][k]
            UserFriendSort.setdefault(i,[]).append(k)
            if len(UserFriendSort[i])>30:
                break
        elif trustcoefficient>mimicthresholdValue:
             mimicthresholdValue = similarCorrcoefMatrix[i][k]
             UsermimicSort.setdefault(i,[]).append(k)
             if len(UserFriendSort[i])>30:
                 break
        else:
            UserFriendSort.setdefault(i, []).append(np.random.randint(1,100))
            if len(UserFriendSort[i])>10:
                break

print(dk)
print(ratingMatrix)
'''
print(ratingMatrix)
print(similarCorrcoefMatrix)

print(UserFriendSort)
print(UsermimicSort)
'''

#print(distMatrix)
#构建用户-物品-关系(密友,普通朋友)三阶因子图(三部图)
'''                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    好友关系矩阵，抽取用户好友
    物品-关系
    信息因子传递信息，调整最大迭代步数，直至算法收敛
    物品列表：labels[]  时间序列:t[]
    UserFriendSort-->labels
    UsermimicSort-->mimiclabels
'''
class RV(object):
       def __init__(self, name, n_opts, labels=[],mimiclabels=[],meta={}, debug=DEBUG_DEFAULT):
           # vars set at construction time
           self.name = name
           self.n_opts = n_opts  # len(labels)==n_opts
           self.labels = labels  #普友列表
           self.mimiclabels=mimiclabels  #密友列表
           self.debug = debug
           self.meta = meta  # metadata: custom data added / manipulated by user
           self.labels=[]
           self.mimiclabels=[]
           self._factors = []
           self._Highfactors= {}
           self._outgoing = None


       def __repr__(self):
           return self.name


       def __hash__(self):
           return hash(self.name)

       def initFactor(self,UsermimicSort,UserFriendSort,similarCorrcoefMatrix):
           #i:用户编号
           for i in range(n):
               self.mimiclabels=[]
               self.labels=[]
               #根据兴趣相似度(Pearson相关系数推荐，陌生人推荐)

               gmUser=UserFavouItem[i][0]
               gsUser=UserFavouItem[i][1]
               RecommendationList.setdefault(i,[]).append(gsUser)  # 取物品列表中的顶部物品推荐给该用户
               RecommendationList.setdefault(i,[]).append(gmUser)

               #根据社交关系推荐(信任好友推荐)
               if i in UsermimicSort.keys() and len(UsermimicSort[i])!=0:
                   for j in range(len(UsermimicSort[i])):
                       for UMSItem in range(10):
                           if UserFavouItem[UsermimicSort[i][j]][UMSItem] not in RecommendationList:
                               RecommendationList.setdefault(i,[]).append(UserFavouItem[UsermimicSort[i][j]][UMSItem])
               elif i in UserFriendSort.keys() and len(UserFriendSort[i])!=0:
                   for k in range(5):
                       for UFSItem in range(10):
                           if UserFavouItem[UserFriendSort[i][k]][UFSItem] not in RecommendationList:
                               RecommendationList.setdefault(i,[]).append(UserFavouItem[UserFriendSort[i][k]][UFSItem])


       #获取所有因子节点
       def get_factors(self):
           return self._factors


       def get_outgoing(self):
           return self._outgoing[:]

       def init_lbp(self):
           self._outgoing = [np.ones(len(self._factors))]
           for i in range(10):
               self._outgoing.append(0.5*len(UserFriendSort[i]))


       def print_messages(self):
           for i, f in enumerate(self._factors):
               print('\t', self, '->', f, '\t', self._outgoing[i-2])

       def n_edges(self):
           return len(self._factors)

       def get_str_label(self, label):
           if type(label) in [str]:
               return self.labels.index(label)
           # assume string otherwise
           return label



if __name__=='__main__':

    datasetName="mydataBase"

    rv=RV(datasetName,UserFriendSort,n,mylabel,{'Mimic':2,'Common':1})#数据集名字,用户列表,物品列表，标签列表,meta
    #rv.create_items()
    rv.initFactor(UsermimicSort,UserFriendSort,similarCorrcoefMatrix)
    rv.init_lbp()
    nR=0
    endTime=time()
    runTime=endTime-beginTime
    print("runTime=%s"%runTime)




