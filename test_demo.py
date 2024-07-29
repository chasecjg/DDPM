import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve

# 生成S形二维数据点
s_curve, _ = make_s_curve(10 ** 4, noise=0.1)  # 10000个数据点

# 创建一个3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制3D S曲线
ax.scatter(s_curve[:, 0], s_curve[:, 1], s_curve[:, 2], s=1, color='blue')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 保存图像
plt.savefig('s_curve_3d.png')
plt.show()
