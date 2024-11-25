import torch

y = torch.randn(10, 1)
y2 = torch.randn(10)

print(y)
print(y2)

print((y-y2).shape)