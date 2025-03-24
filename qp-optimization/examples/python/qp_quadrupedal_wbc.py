def solve_wbc_qp(
    # System state
    q,                  # Joint positions (vector)
    q_dot,              # Joint velocities (vector)
    
    # System dynamics parameters
    H,                  # Mass-inertia matrix (n×n)
    h,                  # Bias forces vector (Coriolis, centrifugal, gravity)
    
    # Contact parameters
    Jc,                 # Contact Jacobian matrix
    Jc_dot,             # Time derivative of contact Jacobian
    mu,                 # Friction coefficient
    
    # Task parameters
    task_jacobians,     # List of task Jacobians [J₁, J₂, ..., Jᵢ]
    task_jacobian_dots, # List of time derivatives of task Jacobians
    v_d,                # List of desired spatial accelerations for each task
    task_weights,       # List of weights for each task [w₁, w₂, ..., wᵢ]
    
    # Desired contact forces (from MPC)
    u_d,                # Desired contact forces
    force_weight,       # Weight matrix for force objective (wf)
    
    # Actuator limits
    S,                  # Actuator selection matrix
    tau_min,            # Minimum joint torque limits
    tau_max,            # Maximum joint torque limits
    
    # Optional solver parameters
    solver_options=None # Dictionary of solver-specific options
):
    """
    Formulates and solves the Whole-Body Control quadratic programming problem.
    
    The QP minimizes:
        ||∑ᵢ wᵢ(J'ᵢq̈ + J̇'ᵢq̇ - v̇ᵈᵢ)||₂ + ||wₑ(uₐ - u)||₂
    
    Subject to constraints:
        Hq̈ + h = J'ᶜu                    # Rigid body dynamics
        Jᶜq̈ = -J̇ᶜq̇                      # Rigid contact constraint
        |f'ᵢ·ê₁| ≤ μf'ᵢ·ê₃, ∀n ∈ {x,y}   # Friction cone constraints
        f'ᵢ·ê₃ > 0, ∀j                   # Normal force positivity
        τₘ ≤ S⁻¹(Hq̈ + h - J'ᶜu) ≤ τₘ     # Torque limits
    
    Args:
        q: Joint positions vector
        q_dot: Joint velocities vector
        H: Mass-inertia matrix
        h: Bias forces vector (Coriolis, centrifugal, gravity)
        Jc: Contact Jacobian matrix
        Jc_dot: Time derivative of contact Jacobian
        mu: Friction coefficient
        task_jacobians: List of task Jacobians for each positioning task
        task_jacobian_dots: List of time derivatives of task Jacobians
        v_d: List of desired spatial accelerations for each task
        task_weights: List of weights for each task
        u_d: Desired contact forces (from MPC)
        force_weight: Weight matrix for force objective
        S: Actuator selection matrix
        tau_min: Minimum joint torque limits
        tau_max: Maximum joint torque limits
        solver_options: Dictionary of solver-specific options
    
    Returns:
        Dictionary containing:
            q_ddot: Optimized joint accelerations
            u: Optimized contact forces
            tau: Computed joint torques
            status: Solver status
    """
    # Import required libraries
    import numpy as np
    import cvxpy as cp
    
    # Get dimensions
    n_joints = len(q)
    n_contacts = Jc.shape[0]
    
    # Define optimization variables
    q_ddot = cp.Variable(n_joints)       # Joint accelerations
    u = cp.Variable(n_contacts)          # Contact forces
    
    # Formulate objective function
    objective_terms = []
    
    # Task objectives
    for i in range(len(task_jacobians)):
        Ji = task_jacobians[i]
        Ji_dot = task_jacobian_dots[i]
        wi = task_weights[i]
        
        # Ji*q_ddot + Ji_dot*q_dot - v_d[i]
        task_error = Ji @ q_ddot + Ji_dot @ q_dot - v_d[i]
        objective_terms.append(wi * cp.norm2(task_error))
    
    # Force objective
    force_error = u - u_d
    objective_terms.append(cp.quad_form(force_error, force_weight))
    
    # Complete objective function
    objective = cp.sum(objective_terms)
    
    # Define constraints
    constraints = []
    
    # Rigid body dynamics constraint: Hq̈ + h = J'ᶜu
    constraints.append(H @ q_ddot + h == Jc.T @ u)
    
    # Contact constraint: Jᶜq̈ = -J̇ᶜq̇
    constraints.append(Jc @ q_ddot == -Jc_dot @ q_dot)
    
    # Friction cone constraints
    # This is a simplified implementation - would need to be expanded for multiple contacts
    # and to properly represent the friction cone constraint from (2d)
    # For each contact point
    n_contact_points = n_contacts // 6  # Assuming 6D contact wrench per contact
    
    for j in range(n_contact_points):
        # Extract contact force components (fx, fy, fz) for this contact
        idx = j * 6
        fx = u[idx]
        fy = u[idx + 1]
        fz = u[idx + 2]
        
        # Friction cone constraints: |fx| ≤ μ*fz, |fy| ≤ μ*fz
        constraints.append(fx <= mu * fz)
        constraints.append(-fx <= mu * fz)
        constraints.append(fy <= mu * fz)
        constraints.append(-fy <= mu * fz)
        
        # Normal force positivity: fz > 0
        constraints.append(fz >= 0.001)  # Small positive value for numerical stability
    
    # Torque limits: τₘ ≤ S⁻¹(Hq̈ + h - J'ᶜu) ≤ τₘ
    tau = S @ (H @ q_ddot + h - Jc.T @ u)
    constraints.append(tau <= tau_max)
    constraints.append(tau >= tau_min)
    
    # Create and solve the optimization problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.OSQP, **solver_options if solver_options else {})
    
    # Extract results
    joint_torques = S @ (H @ q_ddot.value + h - Jc.T @ u.value)
    
    return {
        "q_ddot": q_ddot.value,
        "u": u.value,
        "tau": joint_torques,
        "status": problem.status,
        "objective_value": problem.value
    }