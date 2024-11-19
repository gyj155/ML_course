import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置随机种子以确保结果一致
torch.manual_seed(0)
np.random.seed(0)

# 目标函数: 一个平滑的二维函数
def target_function(x, y):
    return np.sin(x) * np.cos(y)

# 生成训练数据
n_points = 100
x = np.linspace(-3, 3, n_points)
y = np.linspace(-3, 3, n_points)
X, Y = np.meshgrid(x, y)
Z = target_function(X, Y)

# 将数据转换为 PyTorch 张量
train_data = torch.tensor(np.vstack([X.ravel(), Y.ravel()]).T, dtype=torch.float32)
train_labels = torch.tensor(Z.ravel(), dtype=torch.float32).view(-1, 1)

# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()  # 使用Tanh作为非线性激活函数

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义训练函数
def train_model(hidden_size, epochs=1000, lr=0.01):
    model = SimpleNN(hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练模型
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
    
    # 计算并返回预测值
    with torch.no_grad():
        predictions = model(train_data).detach().numpy()
    
    return predictions, loss.item()

# 可视化：逐步增加隐藏层神经元数量并绘制逼近效果
hidden_sizes = [1, 5, 20, 50, 100]  # 隐藏层神经元数量
fig = plt.figure(figsize=(15, 10))
for i, hidden_size in enumerate(hidden_sizes, 1):
    predictions, loss = train_model(hidden_size)
    
    ax = fig.add_subplot(2, 3, i, projection='3d')
    ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.5, rstride=1, cstride=1)
    ax.plot_surface(X, Y, predictions.reshape(n_points, n_points), color='orange', alpha=0.7)
    ax.set_title(f"Hidden Neurons: {hidden_size}, Loss: {loss:.4f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("f(x, y)")

plt.suptitle("Approximating a Smooth Function with Increasing Neurons")
plt.tight_layout()
plt.show()