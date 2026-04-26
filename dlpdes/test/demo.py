import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from torch.autograd.functional import hessian

# ------------------------------
# 参数配置
# ------------------------------
device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
N_interior = 10000
N_boundary = 400
freq = 15
m = 80
lr = 1e-2
epochs = 10000
torch.manual_seed(0)

# ------------------------------
# 精确解和 RHS
# ------------------------------
def u_exact(x):
    return torch.sin(freq * math.pi * x[:,0:1]) * torch.sin(freq * math.pi * x[:,1:2])

def f_rhs(x):
    coef = 2*(freq*math.pi)**2
    return coef * u_exact(x)

# ------------------------------
# 内部点
# ------------------------------
x_in = torch.rand(N_interior,2,device=device)*2 - 1
x_in.requires_grad_(True)

# ------------------------------
# 边界点
# ------------------------------
x_b1 = torch.stack([torch.linspace(-1,1,N_boundary//4), torch.full((N_boundary//4,),-1.0)], dim=1)
x_b2 = torch.stack([torch.linspace(-1,1,N_boundary//4), torch.full((N_boundary//4,), 1.0)], dim=1)
x_b3 = torch.stack([torch.full((N_boundary//4,),-1.0), torch.linspace(-1,1,N_boundary//4)], dim=1)
x_b4 = torch.stack([torch.full((N_boundary//4,), 1.0), torch.linspace(-1,1,N_boundary//4)], dim=1)
x_b = torch.cat([x_b1,x_b2,x_b3,x_b4], dim=0).to(device)
x_b.requires_grad_(True)

# ------------------------------
# 单层 tanh 网络
# ------------------------------
class SingleLayerTanh(nn.Module):
    def __init__(self, hidden=m):
        super().__init__()
        self.a = nn.Parameter(torch.randn(hidden,1)*0.1)
        self.w = nn.Parameter(torch.randn(hidden,2)*0.1)
        self.b = nn.Parameter(torch.randn(hidden,1)*0.1)
    def forward(self,x):
        z = x @ self.w.T + self.b.T
        return (torch.tanh(z) @ self.a).reshape(-1,1)

model = SingleLayerTanh(hidden=m).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_list = []

# ------------------------------
# 训练
# ------------------------------
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # 内部 PDE loss
    u_pred = model(x_in)
    grads = torch.autograd.grad(u_pred.sum(), x_in, create_graph=True)[0]
    u_x = grads[:,0:1]; u_y = grads[:,1:2]
    u_xx = torch.autograd.grad(u_x.sum(), x_in, create_graph=True)[0][:,0:1]
    u_yy = torch.autograd.grad(u_y.sum(), x_in, create_graph=True)[0][:,1:2]
    laplace_u = u_xx + u_yy
    f = f_rhs(x_in)
    loss_in = ((laplace_u + f)**2).mean()
    
    # 边界 loss
    u_b = model(x_b)
    loss_b = (u_b**2).mean()
    
    loss = loss_in + loss_b
    loss.backward()
    optimizer.step()
    
    loss_list.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6e}")

# ------------------------------
# 绘制 Loss 曲线
# ------------------------------
plt.figure()
plt.plot(loss_list)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('PINN Loss')
plt.show()

# ------------------------------
# 平均 L2 norm
# ------------------------------
with torch.no_grad():
    a_avg_l2 = (model.a**2).mean().sqrt().item()
    w_avg_l2 = (model.w**2).mean().sqrt().item()
    b_avg_l2 = (model.b**2).mean().sqrt().item()
    
    print(f"Average L2 norm a: {a_avg_l2:.6f}")
    print(f"Average L2 norm w: {w_avg_l2:.6f}")
    print(f"Average L2 norm b: {b_avg_l2:.6f}")

# ------------------------------
# Hessian 最小特征值（完整 Hessian）
# ------------------------------
def flatten_params(model):
    return torch.cat([model.a.flatten(), model.w.flatten(), model.b.flatten()])

def unflatten_params(params_flat):
    idx_a = m*1
    idx_w = m*2
    a = params_flat[:idx_a].reshape(m,1)
    w = params_flat[idx_a:idx_a+idx_w].reshape(m,2)
    b = params_flat[idx_a+idx_w:].reshape(m,1)
    return a, w, b

def loss_flat(params_flat):
    a, w, b = unflatten_params(params_flat)
    z = x_in @ w.T + b.T
    u = (torch.tanh(z) @ a).reshape(-1,1)
    grads = torch.autograd.grad(u.sum(), x_in, create_graph=True)[0]
    u_x = grads[:,0:1]; u_y = grads[:,1:2]
    u_xx = torch.autograd.grad(u_x.sum(), x_in, create_graph=True)[0][:,0:1]
    u_yy = torch.autograd.grad(u_y.sum(), x_in, create_graph=True)[0][:,1:2]
    laplace_u = u_xx + u_yy
    f = f_rhs(x_in)
    # PDE + 边界 loss
    u_b = (torch.tanh(x_b @ w.T + b.T) @ a).reshape(-1,1)
    return ((laplace_u + f)**2).mean() + (u_b**2).mean()

params_flat = flatten_params(model)
params_flat = params_flat.detach().requires_grad_(True)
H = hessian(loss_flat, params_flat)
eigvals = torch.linalg.eigvals(H)
min_eig = eigvals.real.min().item()
print("Min Hessian eigenvalue (full):", min_eig)

# ------------------------------
# 最后一步梯度 norm（平均 L2 norm）
# ------------------------------
optimizer.zero_grad()
u_pred = model(x_in)
grads = torch.autograd.grad(u_pred.sum(), x_in, create_graph=True)[0]
u_x = grads[:,0:1]; u_y = grads[:,1:2]
u_xx = torch.autograd.grad(u_x.sum(), x_in, create_graph=True)[0][:,0:1]
u_yy = torch.autograd.grad(u_y.sum(), x_in, create_graph=True)[0][:,1:2]
laplace_u = u_xx + u_yy
f = f_rhs(x_in)
loss = ((laplace_u + f)**2).mean() + (model(x_b)**2).mean()
loss.backward()

grad_a_norm = (model.a.grad**2).mean().sqrt().item()
grad_w_norm = (model.w.grad**2).mean().sqrt().item()
grad_b_norm = (model.b.grad**2).mean().sqrt().item()

print(f"Grad avg L2 norm |a|: {grad_a_norm:.6f}, |w|: {grad_w_norm:.6f}, |b|: {grad_b_norm:.6f}")