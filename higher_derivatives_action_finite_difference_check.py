import numpy as np
from fenics import *
from ufl import replace
from higher_derivatives_action import HigherDerivativesAction
from fenics_helper_functions import random_function

n=15
use_amg = False
set_log_level(30)

norm = np.linalg.norm
mesh = UnitSquareMesh(n,n)
V=FunctionSpace(mesh,'CG',1)

m = Function(V)
u = Function(V)
v = Function(V)

def random_smooth_vec():
    return random_function(V, smooth=True)[0]

f = Function(V)
f.vector()[:] = random_smooth_vec()

a0 = exp(m)*inner(grad(u), grad(v))*dx - f*v*dx
a = replace(a0, {v:TestFunction(V)})

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(1.5), boundary)
bc_zero = DirichletBC(V, Constant(0.0), boundary)

Q = exp(m)*inner(grad(u), FacetNormal(mesh))*TestFunction(V)*ds # quantity of interest

QHD = HigherDerivativesAction(Q, a, [bc], m, u)

####

p1 = Function(V)
p2 = Function(V)
p3 = Function(V)
p4 = Function(V)
z = Function(V)

p1.vector()[:] = random_smooth_vec()
p2.vector()[:] = random_smooth_vec()
p3.vector()[:] = random_smooth_vec()
p4.vector()[:] = random_smooth_vec()
z.vector()[:] = random_smooth_vec()

#### Test higher Q's ####

m_vec_a = m.vector()[:].copy()
m.vector()[:] = m_vec_a

q_a = QHD.compute_derivative_of_quantity_of_interest([], None)
q1_a = QHD.compute_derivative_of_quantity_of_interest([p1], None)
q11_a = QHD.compute_derivative_of_quantity_of_interest([p1,p1], None)
q12_a = QHD.compute_derivative_of_quantity_of_interest([p1,p2], None)
q111_a = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p1], None)
q112_a = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p2], None)
q123_a = QHD.compute_derivative_of_quantity_of_interest([p1,p2,p3], None)
q1111_a = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p1,p1], None)
q1112_a = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p1,p2], None)
q1122_a = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p2,p2], None)
q1123_a = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p2,p3], None)
q1234_a = QHD.compute_derivative_of_quantity_of_interest([p1,p2,p3,p4], None)


s = 1e-6
m.vector()[:] = m_vec_a + s * p1.vector()[:].copy()

q_b = QHD.compute_derivative_of_quantity_of_interest([], None)
q1_b = QHD.compute_derivative_of_quantity_of_interest([p1], None)
q11_b = QHD.compute_derivative_of_quantity_of_interest([p1,p1], None)
q111_b = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p1], None)
q1111_b = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p1,p1], None)

q1_diff = (q_b - q_a)/s
q11_diff = (q1_b - q1_a)/s
q111_diff = (q11_b - q11_a)/s
q1111_diff = (q111_b - q111_a)/s

q1_err = norm(q1_diff - q1_a)/norm(q1_diff)
q11_err = norm(q11_diff - q11_a)/norm(q11_diff)
q111_err = norm(q111_diff - q111_a)/norm(q111_diff)
q1111_err = norm(q1111_diff - q1111_a)/norm(q1111_diff)

print('s=', s)
print('q1_err=', q1_err)
print('q11_err=', q11_err)
print('q111_err=', q111_err)
print('q1111_err=', q1111_err)


m.vector()[:] = m_vec_a + s * p2.vector()[:].copy()

q1_b = QHD.compute_derivative_of_quantity_of_interest([p1], None)
q11_b = QHD.compute_derivative_of_quantity_of_interest([p1,p1], None)
q111_b = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p1], None)
q112_b = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p2], None)

q12_diff = (q1_b - q1_a)/s
q112_diff = (q11_b - q11_a)/s
q1112_diff = (q111_b - q111_a)/s
q1122_diff = (q112_b - q112_a)/s

q12_err = norm(q12_diff - q12_a)/norm(q12_diff)
q112_err = norm(q112_diff - q112_a)/norm(q112_diff)
q1112_err = norm(q1112_diff - q1112_a)/norm(q1112_diff)
q1122_err = norm(q1122_diff - q1122_a)/norm(q1122_diff)

print('q12_err=', q12_err)
print('q112_err=', q112_err)
print('q1112_err=', q1112_err)
print('q1122_err=', q1122_err)


m.vector()[:] = m_vec_a + s * p3.vector()[:].copy()

q12_b = QHD.compute_derivative_of_quantity_of_interest([p1,p2], None)
q112_b = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p2], None)

q123_diff = (q12_b - q12_a)/s
q1123_diff = (q112_b - q112_a)/s

q123_err = norm(q123_diff - q123_a)/norm(q123_diff)
q1123_err = norm(q1123_diff - q1123_a)/norm(q1123_diff)

print('q1123_err=', q1123_err)


m.vector()[:] = m_vec_a + s * p4.vector()[:].copy()

q123_b = QHD.compute_derivative_of_quantity_of_interest([p1,p2,p3], None)

q1234_diff = (q123_b - q123_a)/s

q1234_err = norm(q1234_diff - q1234_a)/norm(q1234_diff)

print('q1234_err=', q1234_err)


#### Test higher G's ####

m.vector()[:]= m_vec_a

g_a = QHD.compute_derivative_of_quantity_of_interest([], z)
g1_a = QHD.compute_derivative_of_quantity_of_interest([p1], z)
g11_a = QHD.compute_derivative_of_quantity_of_interest([p1,p1], z)
g12_a = QHD.compute_derivative_of_quantity_of_interest([p1,p2], z)
g111_a = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p1], z)
g112_a = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p2], z)
g123_a = QHD.compute_derivative_of_quantity_of_interest([p1,p2,p3], z)
g1111_a = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p1,p1], z)
g1112_a = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p1,p2], z)
g1122_a = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p2,p2], z)
g1123_a = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p2,p3], z)
g1234_a = QHD.compute_derivative_of_quantity_of_interest([p1,p2,p3,p4], z)

dq_diff = np.dot((q_b - q_a)/s, z.vector())
dq = np.dot(g_a, p1.vector())
g_err = np.abs(dq_diff - dq)/np.abs(dq_diff)
print('g_err=', g_err)


m.vector()[:] = m_vec_a + s * p1.vector()[:].copy()

g_b = QHD.compute_derivative_of_quantity_of_interest([], z)
g1_b = QHD.compute_derivative_of_quantity_of_interest([p1], z)
g11_b = QHD.compute_derivative_of_quantity_of_interest([p1,p1], z)
g111_b = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p1], z)
g1111_b = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p1,p1], z)

g1_diff = (g_b - g_a)/s
g11_diff = (g1_b - g1_a)/s
g111_diff = (g11_b - g11_a)/s
g1111_diff = (g111_b - g111_a)/s

g1_err = norm(g1_diff - g1_a)/norm(g1_diff)
g11_err = norm(g11_diff - g11_a)/norm(g11_diff)
g111_err = norm(g111_diff - g111_a)/norm(g111_diff)
g1111_err = norm(g1111_diff - g1111_a)/norm(g1111_diff)

print('s=', s)
print('g1_err=', g1_err)
print('g11_err=', g11_err)
print('g111_err=', g111_err)
print('g1111_err=', g1111_err)


m.vector()[:] = m_vec_a + s * p2.vector()[:].copy()

g1_b = QHD.compute_derivative_of_quantity_of_interest([p1], z)
g11_b = QHD.compute_derivative_of_quantity_of_interest([p1,p1], z)
g111_b = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p1], z)
g112_b = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p2], z)

g12_diff = (g1_b - g1_a)/s
g112_diff = (g11_b - g11_a)/s
g1112_diff = (g111_b - g111_a)/s
g1122_diff = (g112_b - g112_a)/s

g12_err = norm(g12_diff - g12_a)/norm(g12_diff)
g112_err = norm(g112_diff - g112_a)/norm(g112_diff)
g1112_err = norm(g1112_diff - g1112_a)/norm(g1112_diff)
g1122_err = norm(g1122_diff - g1122_a)/norm(g1122_diff)

print('g12_err=', g12_err)
print('g112_err=', g112_err)
print('g1112_err=', g1112_err)
print('g1122_err=', g1122_err)


m.vector()[:] = m_vec_a + s * p3.vector()[:].copy()

g12_b = QHD.compute_derivative_of_quantity_of_interest([p1,p2], z)
g112_b = QHD.compute_derivative_of_quantity_of_interest([p1,p1,p2], z)

g123_diff = (g12_b - g12_a)/s
g1123_diff = (g112_b - g112_a)/s

g123_err = norm(g123_diff - g123_a)/norm(g123_diff)
g1123_err = norm(g1123_diff - g1123_a)/norm(g1123_diff)

print('g1123_err=', g1123_err)


m.vector()[:] = m_vec_a + s * p4.vector()[:].copy()

g123_b = QHD.compute_derivative_of_quantity_of_interest([p1,p2,p3], z)

g1234_diff = (g123_b - g123_a)/s

g1234_err = norm(g1234_diff - g1234_a)/norm(g1234_diff)

print('g1234_err=', g1234_err)

