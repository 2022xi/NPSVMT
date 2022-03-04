# -*- coding:utf-8 -*-
# from matplotlib import  pyplot as plt

from modelTrain import *
from experiment_method import *


clog = [0.744204286296277, -0.500000000000000, 0.085660192465735]
dataSetList = ["01breast-cancer-wisconsin.txt","02wdbc.txt","03wpbc.txt","04Pima Indian Diabetic.txt","05Heart Disease Data Set.txt"]
bigDataSetList = ["06arcene","07gisette"]
T = 10
"***数据标准化方法选择***"
# stand_method = normalization1
# stand_method = linear2
# stand_method = range3
stand_method = standard4

"***评估方法选择***"
# evaluation_method = holdout_train
evaluation_method = cross_train
# evaluation_method = one_train

"***模型训练方法选择***"
# train_method = model
# train_method = modelThree
train_method = nonModel
# train_method = logModel

"***数据集类型选择***"
read_dataset = read_data
# read_dataset = read_bigData

for M in range(5,6):
    for data_set in dataSetList:
        # data_set = "01breast-cancer-wisconsin.txt"
        jdata, ndata, jtabel, ntabel = read_dataset(data_set, stand_method)
        with open("results/non"+data_set,'a') as f:

            T = 10
            TPsum, FPsum, FNsum, TNsum, iter_num, tsum, run_time, DPtime_sum, SPtime_sum = evaluation_method(jdata, ndata, jtabel, ntabel, T, train_method, clog, M)

            print("\n%d阶逼近，样本扩张%d倍" % (len(clog)-1, M), file=f)
            T = 3
            print("平均训练耗时：%fs， DP耗时：%fs， SP耗时：%fs" % (run_time/T , DPtime_sum/T , SPtime_sum/T ), file=f)

            total = TPsum+FPsum+FNsum+TNsum
            print("%d次测试迭代总数：%d，平均迭代次数：%d" % (T , iter_num, iter_num/T ), file=f)
            try:
                print("%d次测试样本总数：%d，分类正确总数：%d，正确率：%.3f" % (T , total, tsum, tsum/total), file=f)
            except:
                print("参数失控，超出范围", file=f)
            print("%d次总和 TP=%d,FP=%d,FN=%d,TN=%d" % (T , TPsum, FPsum, FNsum, TNsum), file=f)
            try:
                print("Precision=%f,Recall=%f" % (TPsum/(TPsum + FPsum), TPsum/(TPsum + FNsum)), file=f)
            except:
                print("division by zero", file=f)


