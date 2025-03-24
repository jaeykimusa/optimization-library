import cvxpy as cp
import numpy as np

# Problem dimensions
n_q = 18  # Number of joint accelerations
n_u = 12  # Number of contact forces

# Decision variables
ddq = cp.Variable(n_q)  # Joint accelerations (ddot_q)
u = cp.Variable(n_u)    # Contact forces

# Objective function terms
J_i = np.random.randn(6, n_q)     # Jacobian for task i
dot_J_i = np.random.randn(6, n_q) # Time derivative of Jacobian
dot_vd_i = np.random.randn(6)     # Desired spatial acceleration for task i
w_i = np.eye(6)                   # Weight matrix for task i

u_d = np.random.randn(n_u)        # Desired contact forces
w_f = np.eye(n_u)                 # Weight matrix for forces

# Objective function (quadratic terms)
tracking_error = cp.quad_form(J_i @ ddq + dot_J_i @ np.random.randn(n_q) - dot_vd_i, w_i)
force_error = cp.quad_form(u_d - u, w_f)

objective = cp.Minimize(tracking_error + force_error)

# Constraints
H = np.random.randn(n_q, n_q)     # Mass-inertia matrix
h = np.random.randn(n_q)          # Bias vector (Coriolis, gravity, etc.)
J_c = np.random.randn(n_u, n_q)   # Contact Jacobian

# Rigid-body dynamics constraint
constraint_rbd = H @ ddq + h == J_c.T @ u

# Rigid contact constraint
constraint_contact = J_c @ ddq == -np.random.randn(n_u)

# Friction cone constraints (linearized)
mu = 0.5  # Friction coefficient
A_friction = np.vstack([np.eye(n_u), -np.eye(n_u)])
b_friction = np.hstack([mu * u, mu * u])

# Torque limits
S_inv = np.eye(n_q)
tau_m = -10 * np.ones(n_q)
tau_M = 10 * np.ones(n_q)
torque_constraints_lower = S_inv @ (H @ ddq + h - J_c.T @ u) >= tau_m
torque_constraints_upper = S_inv @ (H @ ddq + h - J_c.T @ u) <= tau_M

constraints = [
    constraint_rbd,
    constraint_contact,
    A_friction @ u <= b_friction,
    torque_constraints_lower,
    torque_constraints_upper,
]

# Solve the QP
problem = cp.Problem(objective, constraints)
problem.solve()

# Results
print("Optimal joint accelerations:", ddq.value)
print("Optimal contact forces:", u.value)
