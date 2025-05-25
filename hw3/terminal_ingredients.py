from itertools import product

import cvxpy as cvx

import matplotlib.pyplot as plt

import numpy as np


def generate_ellipsoid_points(M, num_points=100):
    """
    Generate points on a 2-D ellipsoid.
    The ellipsoid is described by the equation
    { x | x.T @ inv(M) @ x <= 1 },
    where inv(M) denotes the inverse of the matrix argument M.
    The returned array has shape (num_points, 2).
    """
    L = np.linalg.cholesky(M)
    theta = np.linspace(0, 2*np.pi, num_points)
    u = np.column_stack([np.cos(theta), np.sin(theta)])
    x = u @ L.T
    return x

def get_M(
    A: np.ndarray,
    rx: float
) -> tuple[np.ndarray, str]:
    n = A.shape[0]
    M_cvx = cvx.Variable((n, n))

    obj = cvx.log_det(M_cvx)
    constraints = [M_cvx >> 0, cvx.bmat([[M_cvx, A@M_cvx], [(A@M_cvx).T, M_cvx]]) >> 0, rx * rx * np.eye(n) - M_cvx >> 0]

    prob = cvx.Problem(cvx.Maximize(obj), constraints)
    prob.solve(cvx.CLARABEL)
    M = M_cvx.value
    status = prob.status

    return M, status

n, m = 2, 1
A = np.array([[0.9, 0.6], [0.0, 0.8]])
B = np.array([[0.0], [1.0]])
Q = np.eye(n)
R = np.eye(m)
P = np.eye(n)
N = 4
rx = 5.0
ru = 1.0

# (d)
M, status = get_M(A, rx)
if status != "optimal":
    raise ValueError("Failed to compute M")
W = np.linalg.inv(M)
print("W", W)
org_points = generate_ellipsoid_points(M)
next_points = (A @ org_points.T).T
entire_state_points = generate_ellipsoid_points(rx * rx * np.eye(n))

plt.figure(figsize=(6, 6))
plt.plot(org_points[:, 0], org_points[:, 1], label='X_T', color='blue')
plt.plot(next_points[:, 0], next_points[:, 1], label='AX_T', color='green')
plt.plot(entire_state_points[:, 0], entire_state_points[:, 1], label='X', color='red')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ellipsoid Plots for Positive Invariant Set')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

# (e)
plt.figure(figsize=(6, 6))
plt.plot(org_points[:, 0], org_points[:, 1], label='X_T', color='blue')
plt.plot(next_points[:, 0], next_points[:, 1], label='AX_T', color='green')
plt.plot(entire_state_points[:, 0], entire_state_points[:, 1], label='X', color='red')

init_x = cvx.Parameter((n,))
x_cvx = cvx.Variable((N + 1, n))
u_cvx = cvx.Variable((N, m))
cost = 0.0
constraints = []
cost += cvx.quad_form(x_cvx[-1], P)
constraints = [x_cvx[0] == init_x]
for k in range(N):
    constraints.append(x_cvx[k + 1] == A @ x_cvx[k] + B @ u_cvx[k])
    constraints.append(cvx.norm(x_cvx[k]) <= rx)
    constraints.append(cvx.norm(u_cvx[k]) <= ru)
    cost += cvx.quad_form(x_cvx[k], Q) + cvx.quad_form(u_cvx[k], R) 
constraints.append(cvx.norm(x_cvx[-1]) <= rx)
constraints.append(cvx.quad_form(x_cvx[-1], W) <= 1)
prob = cvx.Problem(cvx.Minimize(cost), constraints)

T = 15
x0 = np.array([0.0, -4.5])
real_x = [x0]
real_u = []
for time_step in range(T):
    init_x.value = real_x[-1]
    prob.solve(cvx.CLARABEL)
    if prob.status == "infeasible":
        print("Infeasible at time step", time_step)
        break
    x = x_cvx.value
    u = u_cvx.value
    plt.plot(x[:, 0], x[:, 1], "--*", color="k")
    real_x.append(A @ real_x[-1] + B @ u[0]) 
    real_u.append(u[0])
real_x = np.array(real_x)
real_u = np.array(real_u)
plt.plot(real_x[:, 0], real_x[:, 1], "-o")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('MPC Results Overlayed on Ellipsoid Plots')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 6))
plt.plot(real_u)
plt.xlabel('Time Step')
plt.ylabel('Control Input')
plt.title('MPC Control Input Over Time')
plt.grid(True)
plt.show()
