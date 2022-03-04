
# from matplotlib import  pyplot as plt

from modelTrain import *
from experiment_method import *
from phe import paillier

public_key, private_key = paillier.generate_paillier_keypair()
# generate a public key and private key pair
'''[-8,8]
clog = [[0.053213110766315, -0.500000000000000, 0.967553420184348],
[-0.000534755920653, 0.000000000000000, 0.082548292699280, -0.500000000000000, 0.779808255813366],
[0.000009787373558, 0.000000000000000, -0.001388926703873, 0.000000000000000, 0.100770602741314, -0.499999999999998, 0.724273596637649],
[-0.000000211053090, -0.000000000000000, 0.000035001182653, 0.000000000000000, -0.002319898116620, -0.000000000000006, 0.111603724635130, -0.499999999999968, 0.705014713270880],
[0.000000004902826, 0.000000000000000, -0.000000954218236, -0.000000000000000, 0.000074170357428, 0.000000000000016, -0.003155507178483, -0.000000000000260, 0.117774376168773, -0.499999999998936, 0.697834318758961]
]
cexp = [[14.6691699503740,  -61.1329363138352,  -126.6324390509526],
[  -2.8979449855678,   14.6691699503741,   50.1481511319695,  -126.6324390509526],
[   0.490164198699542,  -2.897944985567830, -12.219837521143720,  50.148151131969541,  45.457208766761397],
[  -0.072493383504841,   0.490164198699539,   2.257140063665336, -12.219837521143621, -20.550158114656082,  45.457208766760679],
[   0.009510787087536,  -0.072493383504839,  -0.339868128939972,   2.257140063665184,   5.487518801832681, -20.550158114653662,  -8.508067646118286],
[  -0.001119266700531,   0.009510787087536,   0.043221573842354,  -0.339868128940006,  -1.109113240981101,   5.487518801833717,   3.387643162827909,  -8.508067646122226],
[0.000119240972901, -0.001119266700531, -0.004734534474970, 0.043221573842294, 0.186112974907387, -1.109113240978844, -0.632988588384061, 3.387643162811061, 2.372834380933411],
[-0.000011589511783, 0.000119240972902, 0.000451453014119, -0.004734534475161, -0.027146669374258, 0.186112974914326, 0.045647673346353, -0.632988588469632, -1.651313554245154, 2.372834381094317],
[0.000001034585308, -0.000011589511783, -0.000037580379005, 0.000451453014123, 0.003530873248765, -0.027146669374081, 0.009784276799086, 0.045647673346349, 0.669131028327826, -1.651313554265068, 0.857640645145979]
]
'''
'''[-4,4]
clog = [[0.085660192465735,-0.500000000000000,0.744204286296277],
[-0.001955159086889,-0.000000000000000,0.112473802800219,-0.499999999999999,0.701302509761103],
[0.000075374412293,0.000000000000000,-0.003599691718739,-0.000000000000002,0.121244643503415,-0.499999999999989,0.694619964463433],
[-0.000003339499930,-0.000000000000000,0.000175114143534,0.000000000000004,-0.004520366160939,-0.000000000000048,0.123922969153476,-0.499999999999934,0.693429597507864],
[0.000000158169356,0.000000000000000,-0.000009333286060,-0.000000000000013,0.000254091090164,0.000000000000143,-0.004941576543206,-0.000000000000541,0.124700588320370,-0.499999999999387,0.693203381022893]]

cexp = [[ 1.397531489598849,-3.841078788137727,-0.631021978578594],
[-0.381359930342719, 1.397531489598850,-0.180023456847621,-0.631021978578594],
[0.083163838593508,-0.381359930342719, 0.256998846030741,-0.180023456847618, 1.193830251130378],
[-0.015067978190313,0.083163838593507,-0.113484762514928,0.256998846030743,-1.098452603685765,1.193830251130379],
[0.002330884401662,-0.015067978190313,0.032308178920865,-0.113484762514926,0.528229030951479,-1.098452603685774,0.987178681666977],
[-0.000314168150546,0.002330884401663,-0.006947939837764,0.032308178920855,-0.172539586897262,0.528229030951542,-0.993466249228465,0.987178681666923],
[0.000037483676004,-0.000314168150546,0.001211371945045,-0.006947939837768,0.042642140059172,-0.172539586897276,0.498166598549585,-0.993466249228487,1.000539762734643],
[-0.000004009109228,0.000037483676004,-0.000178330096707,0.001211371945025,-0.008469326040758,0.042642140059294,-0.166298002474634,0.498166598549166,-1.000275250417378,1.000539762734742],
[0.000000388385754,-0.000004009109228,0.000022765900048,-0.000178330096719,0.001405300287103,-0.008469326040682,0.041607855567031,-0.166298002474571,0.500076046838639,-1.000275250417324,0.999984286868939]
]
'''

clog = [[0.123581534871910,-0.721347520444482,1.073659833247951],
[-0.002820698318804,-0.000000000000000,0.162265397529797,-0.721347520444480,1.011765652995331],
]

cexp = []
'''[0.659730166037658,-1.461574115370091,0.933789982539966],
[-0.207249267304880,0.659730166037658,-0.964175873838377,0.933789982539966],
[0.049851852911462,-0.207249267304880,0.488809527484076,-0.964175873838380,1.002158237961400],
[-0.009704296378046,0.049851852911461,-0.164119061180230,0.488809527484073,-1.001144621945221,1.002158237961400],
[0.001585358800289,-0.009704296378045,0.041204441273501,-0.164119061180225,0.500339409667991,-1.001144621945219,0.999962069926365],
[-0.000223023793983,0.001585358800290,-0.008263219555415,0.041204441273525,-0.166739200857839,0.500339409668009,-0.999980115421827,0.999962069926366],
[0.000027540275967,-0.000223023793990,0.001379724739359,-0.008263219555412,0.041678981413779,-0.166739200857870,0.499994289565816,-0.999980115421813,1.000000416604335],
[-0.000003029903412,0.000027540275980,-0.000197358729840,0.001379724739490,-0.008335081735770,0.041678981413922,-0.166665496057328,0.499994289565904,-1.000000216731092,1.000000416604340],
[0.000000300518833,-0.000003029903439,0.000024693255876,-0.000197358729709,0.001389103160442,-0.008335081735149,0.041666476853182,-0.166665496060186,0.500000060901486,-1.000000216730927,0.999999996875621]]
'''
c2List = [clog, cexp]
'''

'''
astr = ["****************对率损失函数逼近****************\n", "****************指数损失函数逼近****************\n"]

dataSetList = ["01breast-cancer-wisconsin.txt","02wdbc.txt","03wpbc.txt","04Pima Indian Diabetic.txt","05Heart Disease Data Set.txt"]

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
# train_method = exp_model
# train_method = old_model
# train_method = model
train_method = modelThree

for data_set in dataSetList:
    jdata, ndata, jtabel, ntabel = read_data(data_set, stand_method)


    with open("results/"+data_set[:-4]+".txt",'w') as f:

        for j in range(2):
            print(astr[j], file=f)
            clist = c2List[j]
            for i in range(len(clist)):
                TPsum, FPsum, FNsum, TNsum, iter_num, tsum, run_time, DPtime_sum, SPtime_sum = evaluation_method(jdata, ndata, jtabel, ntabel, T, train_method, clist[i])

                print("\n%d阶逼近" % (len(clist[i])-1), file=f)
                print("平均训练耗时：%fs， DP耗时：%fs， SP耗时：%fs" % (run_time/T , DPtime_sum/T , SPtime_sum/T ), file=f)

                total = TPsum+FPsum+FNsum+TNsum
                print("%d次测试迭代总数：%d，平均迭代次数：%d" % (T , iter_num, iter_num/T ), file=f)
                try:
                    print("%d次测试样本总数：%d，分类正确总数：%d，正确率：%.3f" % (T , total, tsum, tsum/total), file=f)
                except:
                    print("参数失控，超出范围", file=f)
                print("%d次总和 TP=%d,FP=%d,FN=%d,TN=%d" % (T , TPsum, FPsum, FNsum, TNsum), file=f)
