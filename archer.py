from dolfin import *
import sys

N       = 256
sigma   = 0.9
epsilon = 1
T       = 1

N = int(N)
sigma = float(sigma)
epsilon = float(epsilon)
T = float(T)
h = 1.0 / N
dt = h * 0.0001
iteration = 0

info("Initial parameters: N = {0}; sigma = {1}; epsilon = {2}; h = {3}; dt = {4}".format(N, sigma, epsilon, h, dt))

mesh = UnitIntervalMesh(N)
V = FunctionSpace(mesh, 'CG', 2)

u0 = Expression('(0.375 < x[0] && x[0] < 0.625) ? 1 : 0')

def boundary(x, on_boundary):
  tol = 1E-15
  return abs(x[0]) < tol or \
         abs(x[0] - 1) < tol

bc = DirichletBC(V, Constant(0.0), boundary)

u_1 = interpolate(u0, V)

u = TrialFunction(V)
v = TestFunction(V)

a_K = epsilon*inner(nabla_grad(u), nabla_grad(v))*dx
a_M = u*v*dx
L = u_1*v*dx

M = assemble(a_M)
K = assemble(a_K)
A = M + dt*K
u = Function(V)
t = dt

while t <= T:
  b = M * u_1.vector()
  bc.apply(A, b)
  info("Iteration {0} (Time {1}): Max = {2}; Min = {3}".format(iteration, t, 1, 0))
  solve(A, u.vector(), b, "gmres", "ilu")
  plot(u)
  interactive()
  t+= dt
  u_1.assign(u)
  iteration += 1


