""" 定义XCD最基本的仿真模型 """
import torch as th
from torch.special import bessel_j1 as j1


def TrapeFormFactorBatchPop(paras, qx, qz, DENSITY):
    """
    批量计算多层一维梯形光栅的形状因子
    :param paras: 形状参数列表，包括[下层宽度w1,高度h,左侧壁角正切SWA,右侧壁角正切SWA]
    :param qx, qz: 样品Q矢
    :param DENSITY: 电子密度
    :return: 形状因子
    """
    # 转换成tensor数据格式
    if not isinstance(paras[0],th.Tensor):
        paras = (th.tensor(p,dtype=qx.dtype,device=qx.device) for p in paras)
    # 参数解包，且在后面增加两个维度，pop*k*1*1
    w1, h, slope1, slope2 = (p.view(*p.shape,1,1) for p in paras)
    # 对高度沿第2个维度累积求和
    h_cum = th.cumsum(h,dim=1)
    # 增加 0 元素，并拼接到h_cum上去
    h_zero = th.zeros(h.shape[0],1,1,1,dtype=qx.dtype,device=qx.device)
    h_cum = th.cat([h_zero,h_cum[:,:-1,:,:]],dim=1)
    # 将qx/qz增加两个维度，1*1*m*n
    qx, qz = qx.unsqueeze(0).unsqueeze(0), qz.unsqueeze(0).unsqueeze(0)

    # 执行计算，pop*k*m*n
    term1 = th.exp(-1j * qx * w1/2) * (th.exp(-1j  * (qz - qx * slope2)* h) - 1) / (qx * (qx * slope2 - qz))
    term2 = th.exp( 1j * qx * w1/2) * (th.exp(-1j  * (qz + qx * slope1)* h) - 1) / (qx * (qx * slope1 + qz))
    formfactor = (term1+term2)*th.exp(-1j*qz*h_cum)

    # 返回形状因子,沿第2个维度求和,pop*m*n
    return DENSITY * th.sum(formfactor, dim=1)
