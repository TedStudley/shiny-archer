from dolfin import *
import sys

N           = 256
sigma       = 0.9
epsilon     = 1.25
T           = 1
c           = 0.5
h           = 1.0 / N
dt          = h * 0.0001
iteration   = 0

info("Initial parameters: N = {0}; sigma = {1}; epsilon = {2}; h = {3}; dt = {4}".format(N, sigma, epsilon, h, dt))

mesh = UnitIntervalMesh(N)
V = FunctionSpace(mesh, 'Lagrange', 2)

u0 = Expression('(0.25 < x[0] && x[0] < 0.75) ? 1 : 0')

def boundary(x, on_boundary):
  tol = 1E-15
  return abs(x[0]) < tol or \
         abs(x[0] - 1) < tol

bc = DirichletBC(V, Constant(0.0), boundary)

u_1 = interpolate(u0, V)

u = TrialFunction(V)
v = TestFunction(V)

a = u*v*dx + epsilon*dt*c*inner(nabla_grad(u), nabla_grad(v))*dx

A = assemble(a)
u = Function(V)
t = dt

while t <= T:
  L = u_1*v*dx - epsilon*dt*(1-c)*inner(nabla_grad(u_1), nabla_grad(v))*dx
  b = assemble(L)
  bc.apply(A, b)
  solve(A, u.vector(), b)
  info("Iteration {0} (Time {1}): Min = {2}; Max = {3}".format(iteration, t, u.vector().min(), u.vector().max()))
  plot(u)
  interactive()
  u_1.assign(u)
  t += dt
  iteration += 1
