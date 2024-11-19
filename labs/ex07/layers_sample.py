import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.linspace(-2, 2, 100).reshape(-1, 1)
y = np.sin(5 * x) + 0.2 * np.random.randn(100, 1)  # 目标函数

# 转换为PyTorch张量
x_train = torch.tensor(x, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)

# 定义浅层宽网络
class ShallowWideNet(nn.Module):
    def __init__(self, hidden_size=100):
        super(ShallowWideNet, self).__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义深层窄网络
class DeepNarrowNet(nn.Module):
    def __init__(self, hidden_size=10):
        super(DeepNarrowNet, self).__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

# 训练函数
def train_model(model, epochs=1000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        predictions = model(x_train).detach().numpy()
    
    return predictions, loss.item()

# 训练和可视化
shallow_model = ShallowWideNet(hidden_size=100)
deep_model = DeepNarrowNet(hidden_size=10)

shallow_predictions, shallow_loss = train_model(shallow_model)
deep_predictions, deep_loss = train_model(deep_model)

plt.figure(figsize=(12, 6))
plt.plot(x, y, 'o', label='True Function', markersize=5)
plt.plot(x, shallow_predictions, label=f'Shallow Wide Network (Loss: {shallow_loss:.4f})', color='orange')
plt.plot(x, deep_predictions, label=f'Deep Narrow Network (Loss: {deep_loss:.4f})', color='green')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison of Shallow Wide vs Deep Narrow Network")
plt.legend()
plt.show()