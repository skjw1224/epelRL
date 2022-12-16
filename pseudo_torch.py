
import torch
from utils import FUNs
from env import CstrEnv
from functools import partial
from torchviz import make_dot

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

device = 'cpu'
env = CstrEnv(device)
# dx_eval = System(device)
dx_eval = env.dx_eval
fun = FUNs(device)
jac = fun.fun_jacobian
x, y, prev_u = env.reset()
t = env.time

x0 = torch.tensor([[2.1404, 1.40, 387.34, 386.06, 0., 0.], [2., 1., 400., 390.06, 0., 0.]],
                  dtype=torch.float, requires_grad=True, device=device)
u0 = torch.tensor([[14.19, -1113.5], [14., -1200.]], dtype=torch.float, requires_grad=True, device=device)
t0 = torch.tensor(0, dtype=torch.float, requires_grad=True, device=device)
dt = 20/3600

k10 = 1.287e+12
k20 = 1.287e+12
k30 = 9.043e+9
delHRab = 4.2  # (KJ / MOL)
delHRbc = -11.0  # (KJ / MOL)
delHRad = -41.85  # (KJ / MOL)
param_real = torch.tensor([[k10, k20, k30, delHRab, delHRbc, delHRad],[k10, k20, k30, delHRab, delHRbc, delHRad]],
                          dtype=torch.float, device=device)
p_dim = 6
p_mu0 = param_real
p_sigma0 = torch.zeros([2, p_dim], dtype=torch.float, device=device)
p_eps0 = torch.zeros([2, p_dim], dtype=torch.float, device=device)

m = 4
args = [x0, u0, p_mu0, p_sigma0, p_eps0]
dx, dfdxs = jac(dx_eval, [1, 2, 3], t0, *args)
print(dfdxs['dfdx1'][0])
print(dfdxs['dfdx2'][0])
print(dfdxs['dfdx3'][0])

param_real = torch.tensor([[k10, k20, k30, delHRab, delHRbc, delHRad]],
                          dtype=torch.float, device=device)
p_sigma = torch.zeros([1, p_dim], dtype=torch.float, device=device)
p_eps = torch.zeros([1, p_dim], dtype=torch.float, device=device)
args = [x, prev_u, param_real, p_sigma, p_eps]
graph = dx_eval(t, *args)
# dx, dfdx, _, _ = jac(sys.forward, t, *args)
# print(dfdx)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/pseudo_torch')
writer.add_graph(dx_eval, (t, *args))
writer.close()

y_n = prev_u[0, 0] * y
y_rep = torch.cat([y, y_n])
prev_u_n = torch.tensor([[0.5, -.2]], device=device, requires_grad=True)
prev_u_rep = torch.cat([prev_u, prev_u_n])
cost = env.cost_batch(False, t0, y, prev_u)

hes = fun.hessian
hessian = hes(cost, prev_u)
# TODO: hessian returns strange results for a batch input
# print(hessian)