from dolfin import *
import numpy

n = Constant(128)
sigma = Constant(0.5)
v = Constant(3.0)

mesh = UnitSquareMesh(128, 128)
V = FunctionSpace(mesh, 'CG', 1)

u0 = Expression('((0.375 < x[0] && x[0] < 0.625) && (0.375 < x[1] && x[1] < 0.625)) ? 1 : 0')

def u0_boundary(x):
  tol = 1E-15
  return abs(x[0]) < tol or \
         abs(x[1]) < tol or \
         abs(x[0] - 1) < tol or \
         abs(x[1] - 1) < tol

bc = DirichletBC(V, Constant(0), u0_boundary)

u_1 = project(u0, V)

dt = 0.000001

u = TrialFunction(V)
v = TestFunction(V)
c = Constant(0.6)
epsilon = Constant(0.125)
beta = Constant((0,0.00125))

a = u*v*dx + epsilon*dt*(1-c)*inner(nabla_grad(u), nabla_grad(v))*dx + inner(beta, nabla_grad(u))*v*dx
L = u_1*v*dx-epsilon*dt*c*inner(nabla_grad(u_1), nabla_grad(v))*dx


A = assemble(a)

u = Function(V)
T = 2
t = dt

while t <= T:
  b = assemble(L)
  u0.t = t
  bc.apply(A, b)
  solve(A, u.vector(), b)

  t += dt
  u_1.assign(u)

  u_e = project(u0, V)
  maxdiff = numpy.abs(u_e.vector().array() - u.vector().array()).max()
  print 'Max error, t=%.2f: %-10.3f' % (t, maxdiff)

  b = assemble(L, tensor=b)

  plot(u)
