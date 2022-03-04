import numpy as np
import time
import math
from phe import paillier
# generate a public key and private key pair
public_key, private_key = paillier.generate_paillier_keypair()

MULTIPLE = 1

def horner(n, R, sumy, sumyx, sumx, sumxx):
    hbegin = time.time()
    M = [0]*(n+2)
    for i in range(n):
        for j in range(n):
            a = sumxx[n - i - 1, n - j - 1]
            M[-i-1] *= R
            M[-i-1] += int(a*MULTIPLE)
        M[-i-1] = public_key.encrypt(M[-i-1])

    for i in range(n):
        a = sumx[n - i - 1, 0]
        M[1] *= R
        M[1] += int(a*MULTIPLE)
    M[1] = public_key.encrypt(M[1])

    for i in range(n):
        a = sumyx[n - i - 1, 0]
        M[0] *= R
        M[0] += int(a*MULTIPLE)

    M[0] *= R
    M[0] += int(sumy*MULTIPLE)
    M[0] = public_key.encrypt(M[0])
    # M = list(map(float, M))
    hend = time.time()
    print("horner多项式压缩时间：",hbegin-hend)
    return M

def horner_1(n, R, M):
    hbegin = time.time()
    for i in range(n+2):
        M[i] = private_key.decrypt(M[i])

    sumy = M[0]%R
    M[0] = M[0] //R

    sumyx = []
    sumx = []
    sumxx = []

    for j in range(n):
        sumyx.append(M[0]%R)
        M[0] = M[0] // R

    for j in range(n):
        sumx.append(M[1] % R)
        M[1] = M[1] // R

    for j in range(n):
        temp_list = []
        for i in range(n):
            temp_list.append(M[j+2] % R)
            M[j+2] = M[j+2] // R

        sumxx.append(temp_list)
    sumy /= MULTIPLE
    sumyx = np.mat(sumyx).T/MULTIPLE
    sumx = np.mat(sumx).T/MULTIPLE
    sumxx = np.mat(sumxx)/MULTIPLE
    hend = time.time()
    print("horner多项式解压时间：",hbegin-hend)
    return sumy, sumyx, sumx, sumxx

def horner2(n, R, sumy, sumyx, sumx, sumxx):
    S = 50 # 每组元素个数
    hbegin = time.time()
    print(R)
    symbol = 0
    mlist = []
    M = 0
    num = 0
    for i in range(n):
        for j in range(n):
            num += 1
            a = sumxx[n - i - 1, n - j - 1]
            symbol = symbol<<1
            if a < 0:
                a = -a
                symbol += 1
            else:
                symbol += 0
            M *= R
            M += int(a*MULTIPLE)
            if num == S:
                mlist.append(M)
                M = 0
                num = 0

    for i in range(n):
        num += 1
        a = sumx[n - i - 1, 0]
        symbol = symbol << 1
        if a < 0:
            a = -a
            symbol += 1
        else:
            symbol += 0
        M *= R
        M += int(a*MULTIPLE)
        if num == S:
            mlist.append(M)
            M = 0
            num = 0

    for i in range(n):
        num += 1
        a = sumyx[n - i - 1, 0]
        symbol = symbol << 1
        if a < 0:
            a = -a
            symbol += 1
        else:
            symbol += 0
        M = (M + int(a*MULTIPLE)) * R
        if num == S:
            mlist.append(M)
            M = 0
            num = 0

    a = sumy
    symbol = symbol << 1
    if a < 0:
        a = -a
        symbol += 1
    else:
        symbol += 0
    M += int(a*MULTIPLE)
    mlist.append(M)
    hend = time.time()
    print("聚合时间：",hend-hbegin)
    # M = list(map(float, M))
    print(len(mlist))
    return M, symbol

def Add2(A,B):
    for i in range(len(A)):
            A[i] += B[i]


def nonModel(train_x, train_y, m, clist):

    DPtime = 0
    SPtime = 0

    # SP 计时
    SPbegin = time.time()
    n = train_x[0].shape[0]
    omega = np.mat([0]*n).T
    b = 0

    lam = 0.01
    T = 500
    precision = 1e-3
    cost =10
    C = 1/m
    print("C值：",C,",lam值：",lam)
    t = 0
    fcost = cost
    pcost = 1
    SPend = time.time()
    SPtime += SPend-SPbegin

    # DP 计时
    DPbegin = time.time()
    sumx = np.mat([0] * n).T
    sumxx = np.mat(np.zeros((n, n)))
    sumyx = np.mat([0] * n).T
    sumy = 0
    for i in range(m):
        sumx = sumx + train_x[i]
        sumxx = sumxx + train_x[i] * train_x[i].T
        sumyx = sumyx + train_y[i] * train_x[i]
        sumy = sumy + train_y[i]
    print("DP求参时间：", time.time() - DPbegin)
    sumx_1 = sumx.copy()
    sumxx_1 = sumxx.copy()
    sumyx_1 = sumyx.copy()
    sumy_1 = sumy
    if sumy > 0:
        sumy_1 = 0
    else:
        sumy = 0
        sumy_1 = -sumy_1
    sumx[sumx<0] = 0
    sumx_1[sumx_1>0] = 0
    sumx_1 = -sumx_1

    sumyx[sumyx<0] = 0
    sumyx_1[sumyx_1>0] = 0
    sumyx_1 = -sumyx_1

    sumxx[sumxx<0] = 0
    sumxx_1[sumxx_1>0] = 0
    sumxx_1 = -sumxx_1

    R1 = max(np.max(sumxx), np.max(sumyx), np.max(sumx), sumy)
    R2 = max(np.max(sumxx_1), np.max(sumyx_1), np.max(sumx_1), sumy_1)

    R = int(max(R1,R2) * MULTIPLE) + 10
    # print(R)
    # R = len(R) - 2
    # R = 1<<R

    M = horner(n, R, sumy, sumyx, sumx, sumxx)
    M_1 = horner(n, R, sumy_1, sumyx_1, sumx_1, sumxx_1)
    DPend = time.time()
    DPtime += DPend-DPbegin

    # SP 计时
    SPbegin = time.time()
    sumy, sumyx, sumx, sumxx = horner_1(n, R, M)
    sumy_1, sumyx_1, sumx_1, sumxx_1 = horner_1(n, R, M_1)

    sumy -= sumy_1
    sumyx -= sumyx_1
    sumx -= sumx_1
    sumxx -= sumxx_1
    with open("1.txt",'w') as f:
        print(sumy,sumx,sumyx,sumxx, file=f)
    c1 = C*clist[1]*sumy
    c2 = 2*C*clist[2]*m
    u = C*clist[1]*sumyx
    v = 2*C*clist[2]*sumx
    A = 2*C*clist[2]*sumxx

    while pcost > precision and t < T:
        delta = omega + u + A*omega + b*v
        delta_b = c1 + omega.T*v + c2*b
        omega = omega - lam*delta
        b -= lam*delta_b[0,0]
        cost = np.linalg.norm(np.vstack((omega, b)), 2)
        pcost = abs(cost - fcost)
        fcost = cost
        t += 1

    SPend = time.time()
    SPtime += SPend - SPbegin
    print("迭代次数：",t)
    return omega, b, t, DPtime, SPtime

def logModel(train_x, train_y, m, clist):
    # 对数原损失函数

    DPtime = 0
    SPtime = 0

    # SP 计时
    SPbegin = time.time()
    n = train_x[0].shape[0]
    omega = np.mat([0]*n).T
    b = 0

    lam = 0.01
    T = 500
    precision = 1e-3
    cost =10
    C = 1/m
    print("C值：",C,",lam值：",lam)
    t = 0
    fcost = cost
    pcost = 1
    SPend = time.time()
    SPtime += SPend-SPbegin


    # SP 计时
    SPbegin = time.time()
    while pcost > precision and t < T:
        delta = omega.copy()
        delta_b = 0
        for i in range(m):
            a = train_y[i] * train_x[i]
            expyx = math.exp(-omega.T*a - train_y[i] * b)
            delta = delta - C*a*expyx/(expyx+1)
            delta_b = delta_b - C*train_y[i]*expyx/(expyx+1)
        omega = omega - lam*delta
        b -= lam*delta_b
        cost = np.linalg.norm(np.vstack((omega, b)), 2)
        pcost = abs(cost - fcost)
        fcost = cost
        t += 1

    SPend = time.time()
    SPtime += SPend - SPbegin
    print("迭代次数：",t)
    return omega, b, t, DPtime, SPtime

def modelThree(train_x, train_y, m, clist):
    DPtime = 0
    SPtime = 0

    # SP 计时
    SPbegin = time.time()
    n = train_x[0].shape[0]
    omega = np.mat([0]*n).T
    b = 0

    lam = 0.01
    T = 500
    precision = 1e-3
    cost = 10
    C = 1/m
    print("C值：",C,",lam值：",lam)
    t = 0
    fcost = cost
    pcost = 1
    SPend = time.time()
    SPtime += SPend-SPbegin

    # DP 计时
    DPbegin = time.time()
    sumx = np.mat([0]*n).T
    sumxx = np.mat(np.zeros((n, n)))
    sumyx = np.mat([0]*n).T
    sumy = 0
    for i in range(m):
        sumx = sumx + train_x[i]
        sumxx = sumxx + train_x[i]*train_x[i].T
        sumyx = sumyx - train_y[i] * train_x[i]
        sumy -= train_y[i]
    DPend = time.time()
    DPtime += DPend-DPbegin

    # SP 计时
    SPbegin = time.time()
    while pcost > precision and t < T:

        # SP 计时
        SPbegin = time.time()
        delta = np.mat([0]*(train_x[0].shape[0])).T
        delta_b = 0
        for i in range(m):
            z = omega.T*train_x[i] + b
            z = train_y[i]*z[0,0]
            # print(z)
            cnum = len(clist)
            for j in range(cnum-1):
                # print((cnum-j-1),(cnum-j-1)*clist[j]*(z**(cnum-j-2))*train_y[i]*train_x[i],'delta',delta)
                delta = delta + (cnum-j-1)*clist[j]*(z**(cnum-j-2))*train_y[i]*train_x[i]
                delta_b = delta_b + (cnum-j-1)*clist[j]*(z**(cnum-j-2))*train_y[i]

        # print(delta)
        # print(delta_b)
        delta = omega + C * delta
        delta_b = C*delta_b

        omega = omega - lam*delta
        b -= lam*delta_b
        cost = np.linalg.norm(np.vstack((omega, b)), 2)
        pcost = abs(cost - fcost)
        fcost = cost
        t += 1
        SPend = time.time()

    SPtime += SPend - SPbegin
    print("迭代次数：",t)
    return omega, b, t, DPtime, SPtime

def model(train_x, train_y, m, clist):
    # 指数型泰勒二阶展开

    DPtime = 0
    SPtime = 0

    # SP 计时
    SPbegin = time.time()
    n = train_x[0].shape[0]
    omega = np.mat([0]*n).T
    b = 0

    lam = 0.01
    T = 500
    precision = 1e-3
    cost =10
    C = 1/m
    print("C值：",C,",lam值：",lam)
    t = 0
    fcost = cost
    pcost = 1
    SPend = time.time()
    SPtime += SPend-SPbegin

    # DP 计时
    DPbegin = time.time()
    sumx = np.mat([0]*n).T
    sumxx = np.mat(np.zeros((n, n)))
    sumyx = np.mat([0]*n).T
    sumy = 0
    for i in range(m):
        sumx = sumx + train_x[i]
        sumxx = sumxx + train_x[i] * train_x[i].T
        sumyx = sumyx - train_y[i] * train_x[i]
        sumy -= train_y[i]
    DPend = time.time()
    DPtime += DPend-DPbegin

    # SP 计时
    SPbegin = time.time()
    while pcost > precision and t < T:

        # SP 计时
        SPbegin = time.time()

        delta = sumxx*omega + b*sumx
        delta = omega + C * (delta + sumyx)

        delta_b = omega.T*sumx + m*b
        delta_b = C*(delta_b + sumy)

        omega = omega - lam*delta
        b -= lam*delta_b[0,0]
        cost = np.linalg.norm(np.vstack((omega, b)), 2)
        pcost = abs(cost - fcost)
        fcost = cost
        t += 1
        SPend = time.time()

    SPtime += SPend - SPbegin
    print("迭代次数：",t)
    return omega, b, t, DPtime, SPtime




