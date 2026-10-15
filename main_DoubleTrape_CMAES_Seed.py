# 使用CMA-ES优化双层梯形的噪声数据
# 固定随机数种子

import time, os, random
import numpy as np
# 提前触发 NumPy 的线性代数后端初始化
np.linalg.svd(np.eye(3))
from scipy.io import loadmat,savemat
from deap import base, creator, tools, algorithms, cma
import torch as th
import fitFcns

# 定义精度类型和计算设备
dtype, device = th.float64, th.device('cpu')


# 设置随机数种子（种子范围1——100之间, random.sample(range(1,101),30)）
seeds = [49, 56, 91, 85, 15, 61, 27, 7, 60, 51, 63, 38, 53, 88, 2,
         29, 5, 20, 25, 52, 59, 83, 14, 28, 4, 75, 71, 74, 99, 62]
def setup_seed(seed=42):
    """设置随机数种子，便于复现"""
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True
# end function setup_seed


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
# lb, ub = 0.9*ground_truth, 1.1*ground_truth
# 初始猜测
# paras = np.array()
paras = np.random.uniform(lb, ub)


# 定义保存数据的文件夹、文件名和表头
fpath = r'data/DoubleTrape_CMAES_Seed_20260424/'
if not os.path.exists(fpath): os.makedirs(fpath)
fcsvname = fpath+r'runResult.csv'
columns = ["runnum","err1","err2","err3","err4",
           "err5","err6","err7","cost","L1","time","FEs"]

# 定义仿真函数和误差计算模型
simFcn = lambda x: fitFcns.DoubleTrapeModel(x,Qx,Qz,DENSITY)
# 记一下函数调用次数
def costFcn(x):
    global FEs
    FEs += 1
    # 接口转换，正向模型借助torch库实现，deap库借助numpy库实现
    fitFcns.clip_bounds(x,lb,ub)
    x = th.tensor(x, dtype=dtype, device=device)
    err = fitFcns.CostFunction(x.unsqueeze(0),simFcn,I,'MAE_log')
    # 返回numpy形式的误差结果，且deap要求以元组的形式返回
    return err[0,0].cpu().numpy(),


# 定义适应度类型和个体类型
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 最小化问题
creator.create("Individual", list, fitness=creator.FitnessMin)

# 设置使用裁剪方法的CMA-ES

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
# stats.register("std", np.std)
stats.register("min", np.min)
# stats.register("max", np.max)

# 开始循环
for i in range(30):
    # 设置随机数种子
    setup_seed(seeds[i])
    print(f'Seed: {seeds[i]}...')

    FEs = 0
    toolbox = base.Toolbox()
    # 初始化策略,这个lambda_控制种群数量
    strategy = cma.Strategy(centroid=paras, sigma=0.9, lambda_=200)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    toolbox.register("evaluate", costFcn)
    # hof 用于记录最佳个体
    hof = tools.HallOfFame(1)

    tic = time.time()
    # The CMA-ES algorithm converge with good probability with those settings
    # verbose 设置成 False 时不向命令行中输出迭代信息
    pop, log = algorithms.eaGenerateUpdate(toolbox, ngen=350, stats=stats, halloffame=hof, verbose=False)
    toc = time.time()-tic
    iter_gen, iter_min, iter_avg = np.array(log.select('gen','min','avg'))
    err = np.array(hof[0])-ground_truth
    L1 = np.linalg.norm(err[:5],ord=1)
    # print(f'Error: {err}')
    print(f'Runnum: {i:2d}, Cost: {np.min(iter_min):.4f}, L1: {L1:.4f}, Time: {toc}s')

    # 写入数据
    row_data = [i+1, *err, np.min(iter_min), L1, toc, FEs]
    fitFcns.append_row(fcsvname, row_data, columns)
    savemat(fpath + f'{i+1}.mat',
            {'x':np.array(hof[0]),'err':err,'cost':iter_min})

print('OK!')

