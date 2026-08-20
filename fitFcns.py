# 汇聚一些拟合用到的函数
import XCD, os
import torch as th
import pandas as pd


def append_row(filename,  row_data, columns):
    """追加一行到 CSV 文件，row_data 为 list/tuple"""
    df = pd.DataFrame([row_data], columns=columns)

    if not os.path.exists(filename):
        # 第一次运行：写入表头
        df.to_csv(filename, mode="w", header=True, index=False)
    else:
        # 之后运行：追加，不写表头
        df.to_csv(filename, mode="a", header=False, index=False)


def CostFunction(paras, model, I, error_type='MSE'):
    """
    定制的误差评价函数--多个体同时计算误差
    :param paras: 待拟合参数
    :param model: 物理模型，调用方式为 model(paras)
    :param I: 实验光强数据 (m*n)
    :param error_type: 要计算的误差类型（'MAE'|'MSE'|'MAE_log'|'Chi'）
    :return: 误差数值 (pop,1)
    """
    # 根据拟合参数计算仿真光强
    I_sim = model(paras)  # pop*m*n

    # 扩展I的维度以匹配I_sim
    I_expanded = I.unsqueeze(0)  # 1*m*n

    if error_type == 'Chi':
        residual_err = th.sum(((I_sim - I_expanded) ** 2) / th.abs(I_sim), dim=(1, 2)) / I.numel()
    elif error_type == 'MAE_log':
        residual_err = th.sum(th.abs(th.log2(I_sim+1) - th.log2(I_expanded+1)), dim=(1, 2)) / (I.numel() - 1)
    elif error_type == 'MAE':
        residual_err = th.sum(th.abs(I_sim - I_expanded), dim=(1, 2)) / (I.numel() - 1)
    elif error_type == 'MSE':
        residual_err = th.sqrt(th.sum((I_sim - I_expanded) ** 2, dim=(1, 2)) / (I.numel() - 1))
    else:
        raise ValueError(f"Unknown cost function type: {error_type}")

    return residual_err.unsqueeze(-1)  # 确保返回(pop,1)而不是(pop,)
# end function CostFunction


def clip_bounds(individual, lb, ub):
    """将个体裁剪到边界内"""
    for i in range(len(individual)):
        individual[i] = max(lb[i], min(ub[i], individual[i]))
    return individual



def DoubleTrapeModel(paras, Qx, Qz, DENSITY):
    """
    两层一维梯形光栅光强拟合模型
    :param paras: 待拟合参数，应能解析出高度、宽度、Is、Ibk等数据
    :param Qx, Qz: 样品坐标系的Q矢
    :param DENSITY: 电子密度序列
    :return: 散射光强I
    """

    # 参数解包,像paras[:,i]这种会默认降成一维，所以添加2个None使其变成pop*1*1
    width, height = paras[:,:3], paras[:,3:5]
    Is, Ibk = (paras[:,i,None,None] for i in range(5, paras.shape[1]))
    # 计算形状因子,形状上 formparas = (pop*k, ...)
    slope1 = 0.5*(width[:,:-1]-width[:,1:])/height
    formparas = (width[:,:-1], height, slope1, slope1)

    # size(formfactor) = pop*m*n
    formfactor = XCD.TrapeFormFactorBatchPop(formparas, Qx, Qz, DENSITY)
    # 返回计算光强, pop*m*n
    return Is*th.abs(formfactor)**2+Ibk
# end function DoubleTrapeModel

