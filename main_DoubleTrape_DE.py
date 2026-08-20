# 使用DE优化双层梯形的噪声数据

import fitFcns, time, os
import torch as th
import numpy as np
from scipy.io import loadmat,savemat
from scipy.optimize import differential_evolution as DE
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 定义精度类型和计算设备
dtype, device = th.float64, th.device('cpu')

# 加载数据
tmp = loadmat(r"expdata\DoubleTrapeSimDataNoise_260424.mat")
keys = ('I', 'Qx1', 'Qz1')
I, Qx, Qz = (th.tensor(tmp[key], dtype=dtype, device=device) for key in keys)

# 设置材料电子密度
DENSITY = 1.0

# 设置真值和上下界
ground_truth = np.array([81, 55, 24, 83, 9, 0.01, 1],dtype=np.float64)
lb = np.array([70, 35, 5, 65, 1, 0.001, 0],dtype=np.float64)
ub = np.array([100, 75, 45, 105, 30, 0.1, 10],dtype=np.float64)
bounds = [(l,u) for l,u in zip(lb,ub)]

# lb, ub = 0.9*ground_truth, 1.1*ground_truth
# 初始猜测
# paras = np.array()
# paras = np.random.uniform(lb, ub)


# 定义保存数据的文件夹、文件名和表头
fpath = r'data/DoubleTrape_DE_20260424/'
if not os.path.exists(fpath): os.makedirs(fpath)
fcsvname = fpath+r'runResult.csv'
columns = ["runnum","err1","err2","err3","err4",
           "err5","err6","err7","cost","L1","time","FEs"]

# 定义仿真函数和误差计算模型
simFcn = lambda x: fitFcns.DoubleTrapeModel(x,Qx,Qz,DENSITY)
def costFcn(x):
    #
    global FEs
    FEs += 1
    # 接口转换，正向模型借助torch库实现，deap库借助numpy库实现
    fitFcns.clip_bounds(x,lb,ub)
    x = th.tensor(x, dtype=dtype, device=device)
    err = fitFcns.CostFunction(x.unsqueeze(0),simFcn,I,'MAE_log')
    # 返回numpy形式的误差结果，且deap要求以元组的形式返回
    return err[0,0].cpu().numpy()


for i in range(30):
    cost_list, FEs = [], 0
    callback = lambda x,_: cost_list.append(costFcn(x))

    tic = time.time()
    # polish=True 调用局部算法精修
    res = DE(costFcn, bounds, strategy='best1bin', maxiter=350, popsize=200, tol=1e-12,
             init='random',callback=callback, recombination=0.9, polish=False)
    toc = time.time()-tic
    err = res.x-ground_truth
    L1 = np.linalg.norm(err[:5],ord=1)
    # print(f'Error: {err}')
    print(f'Cost: {res.fun:.4f}, L1: {L1:.4f}, Time: {toc}s')

    # 写入数据
    row_data = [i+1, *err, res.fun, L1, toc,FEs]
    fitFcns.append_row(fcsvname, row_data, columns)
    savemat(fpath + f'{i+1}.mat',
            {'x':res.x,'err':err,'cost':cost_list})

print('OK!')

