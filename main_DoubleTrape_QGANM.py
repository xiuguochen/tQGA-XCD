# 使用QGA+NM优化双层梯形的噪声数据

import fitFcns, time, os
import torch as th
from scipy.io import loadmat, savemat
from scipy.optimize import minimize
from optim_toolkit_torch import SelectionQA
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
ground_truth = [81, 55, 24, 83, 9, 0.01, 1]
lb = [70, 35, 5, 65, 1, 0.001, 0]
ub = [100, 75, 45, 105, 30, 0.1, 10]
# lb, ub = [0.9*gt for gt in ground_truth], [1.1*gt for gt in ground_truth]
# 参数空间的分辨精度要求
precision = [0.1]*5+[0.0005, 0.01]


# 定义仿真函数和误差计算模型
simFcn = lambda x: fitFcns.DoubleTrapeModel(x,Qx,Qz,DENSITY)
# costFcn = lambda x,knowns: fitFcns.CostFunction(x,*knowns)
def costFcn(x,knowns):
    global FEs
    FEs += 1
    return fitFcns.CostFunction(x,*knowns)
def costFcn1(x):
    # 这个误差函数是给后续的局部优化算法使用的，需要做数据转换
    global FEs1
    FEs1 += 1
    # 转成tensor数据，并增加一个维度
    x = th.tensor(x,dtype=dtype,device=device)
    x = x.unsqueeze(0)
    # 计算误差，返回时去掉增加的维度
    cost = fitFcns.CostFunction(x,simFcn,I,'MAE_log')
    return cost.cpu().numpy()[0,0]

# 定义保存数据的文件夹、文件名和表头
fpath = r'data/DoubleTrape_20260424/'
if not os.path.exists(fpath): os.makedirs(fpath)
fcsvname = fpath+r'runResult.csv'
columns = ["err_type","selection_rate","runnum","err1","err2","err3","err4",
           "err5","err6","err7","cost1","L1","time1","cost2","L1_2","time2","FE1","FE2"]


# 跑多次设置
runNums = 30
# selection_rate_list = th.arange(0.5,1.01,0.05).tolist()
selection_rate_list = (0.50,)
# err_type_list = ('Chi','MSE','MAE','MAE_log')
err_type_list = ('MAE_log',)

# 开始进入循环
with th.inference_mode():
    for err_type in err_type_list:
        knowns = (simFcn, I, err_type)
        for selection_rate in selection_rate_list:
            for runnum in range(runNums):
                # 记录函数调用次数的FEs
                # global FEs, Fes1
                FEs, FEs1 = 0, 0
                # 初始化QGA实例
                qga = SelectionQA(fun=costFcn, n_dim=len(lb), lb=lb, ub=ub, precision=precision,
                                  selection_rate=selection_rate,pop_size=200,max_iter=50,
                                  theta=0.1*th.pi,mutation_rate=0.001,fun_known=knowns,device=device)
                tic = time.time()
                s_qga, f_qga = qga.evolve()
                toc = time.time()-tic
                err = th.tensor(s_qga)-th.tensor(ground_truth)
                L1 = th.norm(err[:5],1).cpu().numpy()

                # 输出结果
                print(f'err_type: {err_type}, selection_rate: {selection_rate:.2f},'
                      f' round: {runnum}, Cost: {f_qga:.4f}, L1: {L1:.4f}, Time: {toc}s')
                # print(s_qga)
                # print(f'Error: {err}')

                # # 如果只使用这种梯度算法，无法收敛
                # s_qga = np.random.uniform(lb,ub)
                # 设置回调函数以记录迭代过程中的损失
                cost2_list = []
                callback = lambda x: cost2_list.append(costFcn1(x))

                # simplex = np.vstack([s_qga] + [s_qga+0.02*e for e in np.eye(len(lb))])
                opts = {'maxiter':300,'fatol':1e-16,'adaptive':True}
                tic = time.time()
                res = minimize(costFcn1, s_qga, method='Nelder-Mead', callback=callback, options=opts)
                # res = least_squares(costFcn1, s_qga, method='trf', bounds=(lb, ub))
                toc2 = time.time()-tic
                err2 = res.x-ground_truth
                L1_2 = th.norm(th.tensor(err2[:5]),1).cpu().numpy()
                print(f'Cost: {f_qga:.4f} --> {res.fun:.4f}, L1: {L1:.4f} --> {L1_2:.4f}, Time: {toc2}s')

                # 写入数据
                row_data = [err_type,selection_rate,runnum+1,*err2,f_qga,L1,toc,res.fun,L1_2,toc2,FEs,FEs1-300]
                fitFcns.append_row(fcsvname, row_data, columns)
                savemat(fpath + f'{err_type}_{selection_rate*100:.0f}_{runnum+1}.mat',
                        {'x':res.x, 'err':err, 'err2':err2, 'cost1':qga.best_fitness_history, 'cost2':cost2_list})

print('\nOK!')


