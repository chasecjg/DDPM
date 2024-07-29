import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve  # 生成S形二维数据点 https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_s_curve.html
import torch
import torch.nn as nn
from tqdm import tqdm

from train import MLPDiffusion

## ----------------------------- 1、生成数据，(10000, 2)的数据点集，组成一个S形 ----------------------------- ##
s_curve, _ = make_s_curve(10 ** 4, noise=0.1)  # 10000个数据点
s_curve = s_curve[:, [0, 2]] / 10.0
print("shape of s:", np.shape(s_curve))
dataset = torch.Tensor(s_curve).float()

## ----------------------------- 2、确定超参数的值 ----------------------------- ##
# 采样时间步总长度 t
num_steps = 100
 
# 制定每一步的beta
betas = torch.linspace(-6, 6, num_steps)
betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
 
# 计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)


def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    """
    从x[t]采样t-1时刻的重构值x[t-1]，根据论文中的采样公式计算单步的采样
    :param model:
    :param x: x[T]
    :param t:
    :param betas:
    :param one_minus_alphas_bar_sqrt:
    :return:
    """
    ## 1) 求出 bar_u_t
    t = torch.tensor([t])
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t] # 这里先计算采样公式中的一部分参数，方便后面表示，看不懂的可以直接对着论文公式看
    # 送入U-Net模型，得到t时刻的随机噪声预测值 eps_theta
    eps_theta = model(x, t)
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))
 
    ## 2) 得到 x[t-1]
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return sample

def p_sample_loop(model, noise_x_t, n_steps, betas, one_minus_alphas_bar_sqrt):
    """
    从x[T]恢复x[T-1]、x[T-2]|...x[0] 的循环
    :param model:
    :param shape:数据集的形状，也就是x[T]的形状
    :param n_steps:
    :param betas:
    :param one_minus_alphas_bar_sqrt:
    :return: x_seq由x[T]、x[T-1]、x[T-2]|...x[0]组成, cur_x是从噪声中生成的图片
    """
    # 得到噪声x[T]
    cur_x = noise_x_t # 初始化当前的x为噪声x[T]
    x_seq = [noise_x_t] # 初始化x序列为第一个元素为x[T],也就是纯噪声
    # 从x[T]恢复x[T-1]、x[T-2]|...x[0]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq, cur_x

# 1) 加载训练好的diffusion model
model = MLPDiffusion(num_steps)
model.load_state_dict(torch.load('./checkpoints_cpu/model_3900.pth'))

# 2) 生成随机噪声x[T]
noise_x_t = torch.randn(dataset.shape)

# 3) 根据随机噪声逆扩散为x[T-1]、x[T-2]|...x[0] + 图片x[0]
x_seq, cur_x = p_sample_loop(model, noise_x_t, num_steps, betas, one_minus_alphas_bar_sqrt)

# 4) 绘制并保存图像
def plot_samples(x_seq, cur_x):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # 绘制 x_seq
    for i, x in enumerate(x_seq):
        if i % 10 == 0:  # 每10个时间步绘制一次
            ax[0].scatter(x.detach().numpy()[:, 0], x.detach().numpy()[:, 1], label=f'Step {i}', alpha=0.5)
    ax[0].legend()
    ax[0].set_title('x_seq')
    
    # 绘制 cur_x
    ax[1].scatter(cur_x.detach().numpy()[:, 0], cur_x.detach().numpy()[:, 1], color='red')
    ax[1].set_title('cur_x')
    
    plt.savefig('samples_plot.png')
    plt.show()

plot_samples(x_seq, cur_x)
