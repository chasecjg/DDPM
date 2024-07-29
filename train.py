import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve  # 生成S形二维数据点 https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_s_curve.html
import torch
import torch.nn as nn
from tqdm import tqdm

## ----------------------------- 1、生成数据，(10000, 2)的数据点集，组成一个S形 ----------------------------- ##
s_curve, _ = make_s_curve(10 ** 4, noise=0.1)  # 生成10000个数据点，形状为S形并且带有噪声，shape为(10000,3)，形状是3维的
s_curve = s_curve[:, [0, 2]] / 10.0 # 选择数据的第一列和第三列，并进行缩放。
print("shape of s:", np.shape(s_curve))
dataset = torch.Tensor(s_curve).float()

## ----------------------------- 2、确定超参数的值 ----------------------------- ##
# 采样时间步总长度 t
num_steps = 100
 
# 制定每一步的beta
betas = torch.linspace(-6, 6, num_steps) # 在-6到6之间生成100个等间距的值
betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5 # 将betas缩放到合适的范围
 
# 计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
alphas = 1 - betas # 计算每一步的alpha值
alphas_prod = torch.cumprod(alphas, 0) # 每个t时刻的alpha值的累积乘积
# alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod) # 计算累积乘积的平方根
one_minus_alphas_bar_log = torch.log(1 - alphas_prod) # 计算1减去累积乘积的对数
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod) # 计算1减去累积乘积的平方根


## ----------------------------- 3、确定扩散前向过程任意时刻的采样值 x[t]： x[0] + t --> x[t] ----------------------------- ##此代码并未使用这个
def q_x(x_0, t):
    """
    x[0] + t --> x[t]
    :param x_0:初始数据
    :param t:任意时刻
    :return:
    """
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    x_t = alphas_t * x_0 + alphas_1_m_t * noise
    return x_t

## ----------------------------- 4、编写求逆扩散过程噪声的模型U-Net（这里使用的是MLP模拟U-Net，官方使用的是U-Net） x[t] + t --> noise_predict----------------------------- ##预测噪声
class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, num_units=128):
        super(MLPDiffusion, self).__init__()
 
        self.linears = nn.ModuleList(
            [
                nn.Linear(2, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 2),
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )
 
    def forward(self, x, t):
        #  x = x[0]
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)
        x = self.linears[-1](x)
 
        return x
    

## ----------------------------- 损失函数 = 真实噪声eps与预测出的噪声noise_predict 之间的loss ----------------------------- ##
def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    """对任意时刻t进行采样计算loss"""
    batch_size = x_0.shape[0]
 
    # 对一个batchsize样本生成随机的时刻t, t的形状是torch.Size([batchsize, 1])
    t = torch.randint(0, n_steps, size=(batch_size // 2,)) # 随机生成时间步t，一半时间
    t = torch.cat([t, n_steps - 1 - t], dim=0) # 创建对称的时间步
    t = t.unsqueeze(-1) # 添加一个维度，使t的形状为(batch_size, 1)
 
    ## 1) 根据 alphas_bar_sqrt, one_minus_alphas_bar_sqrt --> 得到任意时刻t的采样值x[t]
    # x0的系数
    a = alphas_bar_sqrt[t] # 获取时间步t对应的alphas_bar_sqrt值
    # 噪声eps的系数
    aml = one_minus_alphas_bar_sqrt[t] # 获取时间步t对应的one_minus_alphas_bar_sqrt值
    # 生成生成与x_0形状相同的随机噪声e
    e = torch.randn_like(x_0)
    # 计算任意时刻t的采样值
    x = x_0 * a + e * aml
 
    ## 2) x[t]送入U-Net模型，得到t时刻的随机噪声预测值，这里是用UNet直接预测噪声，输入网络的参数是加上噪声的图像和时间t，网络返回预测所加的噪声
    output = model(x, t.squeeze(-1))
 
    ## 3)计算真实噪声eps与预测出的噪声之间的loss
    loss = (e - output).square().mean()
    return loss



## ----------------------------- 训练模型 ----------------------------- ##

if __name__ == "__main__":
    print('Training model...')
    batch_size = 128
    num_epoch = 4000
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = MLPDiffusion(num_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for t in tqdm(range(num_epoch),desc="Traing epoch"):
        for idx, batch_x in enumerate(dataloader):
            loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
    
        if (t % 100 == 0):
            print(loss)
            torch.save(model.state_dict(), 'model_{}.pth'.format(t))