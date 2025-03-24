import cvxpy as cp
import numpy as np

def solve_wbc_qp(A_cG, h_Ga, A_dot_cG_q_dotP, L_S, L_P, q_dotP, q_ddot_U_d, w1=0.7, w2=0.3):
    
    # Problem dimensions
    n = 18  # Number of joint accelerations
    c = 12  # Number of contact forces
    
    # Decision variables
    q_ddot = cp.Variable(n)
    f_contact = cp.Variable(c)

    # Objective function: minimize tracking errors
    objective = cp.Minimize(
        w1 * cp.norm(A_cG @ q_ddot - h_Ga + A_dot_cG_q_dotP, 2) +
        w2 * cp.norm(q_ddot - q_ddot_U_d, 2)
    )

    # Constraints
    q_dotS = -np.linalg.inv(L_S) @ (L_P @ q_dotP)
    constraints = [
        q_ddot[:L_S.shape[0]] == q_dotS,
        f_contact >= 0,  # Non-negative contact forces
        cp.norm(f_contact) <= 100  # Force magnitude limit (example)
    ]
    
    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    
    # Check solver status
    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise ValueError(f"Solver failed with status: {prob.status}")
    
    # Define joint names and contact force labels
    joint_names = [
        'Left hip X', 'Left hip Y', 'Left hip Z',
        'Left knee', 'Left ankle X', 'Left ankle Y',
        'Right hip X', 'Right hip Y', 'Right hip Z',
        'Right knee', 'Right ankle X', 'Right ankle Y',
        'Torso X', 'Torso Y', 'Torso Z',
        'Left shoulder X', 'Left shoulder Y', 'Left elbow'
    ]

    contact_force_labels = [
        'Left foot force X', 'Left foot force Y', 'Left foot force Z',
        'Right foot force X', 'Right foot force Y', 'Right foot force Z',
        'Left hand force X', 'Left hand force Y', 'Left hand force Z',
        'Right hand force X', 'Right hand force Y', 'Right hand force Z'
    ]
    
    # Return results with labeled outputs
    joint_accelerations = {name + ' (rad/s²)': val for name, val in zip(joint_names, q_ddot.value)}
    contact_forces = {label + ' (N)': val for label, val in zip(contact_force_labels, f_contact.value)}
    
    return joint_accelerations, contact_forces, result


# Test the function with reasonable assumed values
if __name__ == "__main__":
    # Centroidal momentum matrix (6x18), approximating physical constraints
    A_cG = np.eye(6, 18)
    
    # Desired momentum rate change — small values representing balance corrections
    h_Ga = np.array([0.1, -0.05, 0.2, 0, 0.05, -0.1])
    
    # Momentum derivative, small to reflect steady-state adjustments
    A_dot_cG_q_dotP = np.zeros(6)
    
    # Selection matrix for constrained velocities
    L_S = np.eye(6)
    
    # Mapping joint velocities to constraints
    L_P = np.random.uniform(-0.1, 0.1, (6, 18))
    
    # Initial joint velocities, starting at rest
    q_dotP = np.zeros(18)
    
    # Desired joint accelerations, targeting small controlled movements
    q_ddot_U_d = np.ones(18) * 0.05

    joint_accel_vals, contact_force_vals, cost_val = solve_wbc_qp(
        A_cG, h_Ga, A_dot_cG_q_dotP, L_S, L_P, q_dotP, q_ddot_U_d
    )

    # Output the results with names
    print("Optimized joint accelerations:")
    for k, v in joint_accel_vals.items():
        print(f"{k}: {v}")

    print("\nOptimized contact forces:")
    for k, v in contact_force_vals.items():
        print(f"{k}: {v}")

    print("\nOptimal cost:", cost_val)

# Let me know if you’d like me to refine anything further! 🚀
