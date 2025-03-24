import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Import the solver function (assuming it's in a file called wbc_solver.py)
from qp_quadrupedal_wbc import solve_wbc_qp

def run_wbc_example():
    """
    Example usage of the Whole-Body Control QP solver for a simple humanoid robot.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define robot dimensions
    n_joints = 18  # Number of joints (typical for a simplified humanoid)
    n_contacts = 12  # 2 feet with 6D contact wrench each
    n_tasks = 3  # CoM task, torso orientation, and hands position
    
    # System state (initial configuration)
    q = np.zeros(n_joints)  # Joint positions (neutral stance)
    q_dot = np.zeros(n_joints)  # Joint velocities (stationary)
    
    # System dynamics parameters
    # Mass-inertia matrix (typically dense for a humanoid)
    H = np.eye(n_joints) * 5.0 + np.random.rand(n_joints, n_joints) * 0.5
    H = (H + H.T) / 2.0  # Make symmetric
    
    # Bias forces (gravity, Coriolis, centrifugal)
    h = np.ones(n_joints) * 9.81 * 0.5  # Simplified gravity effect
    
    # Contact parameters
    # Contact Jacobian (maps joint velocities to contact point velocities)
    Jc = np.random.rand(n_contacts, n_joints) * 0.5
    Jc_dot = np.zeros((n_contacts, n_joints))  # Assume stationary contacts for simplicity
    mu = 0.7  # Typical friction coefficient
    
    # Task parameters
    task_jacobians = []
    task_jacobian_dots = []
    v_d = []
    task_weights = []
    
    # CoM task (position control of center of mass)
    J_com = np.random.rand(3, n_joints) * 0.8
    task_jacobians.append(J_com)
    task_jacobian_dots.append(np.zeros((3, n_joints)))
    v_d.append(np.array([0.0, 0.0, 0.0]))  # Keep CoM stationary
    # FIXED: Using scalar weights instead of matrices
    task_weights.append(10.0 * np.identity(3))  # High priority on CoM
    
    # Torso orientation task
    J_torso = np.random.rand(3, n_joints) * 0.8
    task_jacobians.append(J_torso)
    task_jacobian_dots.append(np.zeros((3, n_joints)))
    v_d.append(np.array([0.0, 0.0, 0.0]))  # Keep torso upright
    task_weights.append(5.0 * np.identity(3))  # Medium priority
    
    # Hand position task
    J_hand = np.random.rand(3, n_joints) * 0.8
    task_jacobians.append(J_hand)
    task_jacobian_dots.append(np.zeros((3, n_joints)))
    v_d.append(np.array([0.1, 0.0, 0.0]))  # Move hand forward
    task_weights.append(1.0 * np.identity(3))  # Lower priority
    
    # Desired contact forces (from MPC)
    u_d = np.zeros(n_contacts)
    u_d[2] = 50.0  # Normal force on first foot (z-component)
    u_d[8] = 50.0  # Normal force on second foot (z-component)
    force_weight = 0.1 * np.eye(n_contacts)  # FIXED: Ensure this is PSD
    
    # Actuator limits
    S = np.eye(n_joints)  # All joints are actuated
    tau_min = np.ones(n_joints) * -100.0  # Min torque
    tau_max = np.ones(n_joints) * 100.0   # Max torque
    
    # Solver options
    solver_options = {
        'verbose': True,  # Changed to True to see solver output
        'max_iter': 5000,  # Increased iterations
        'eps_abs': 1e-4,
        'eps_rel': 1e-4
    }
    
    # Solve the QP problem
    print("Solving WBC quadratic programming problem...")
    result = solve_wbc_qp(
        q=q,
        q_dot=q_dot,
        H=H,
        h=h,
        Jc=Jc,
        Jc_dot=Jc_dot,
        mu=mu,
        task_jacobians=task_jacobians,
        task_jacobian_dots=task_jacobian_dots,
        v_d=v_d,
        task_weights=task_weights,
        u_d=u_d,
        force_weight=force_weight,
        S=S,
        tau_min=tau_min,
        tau_max=tau_max,
        solver_options=solver_options
    )
    
    # Print results
    print(f"Solver status: {result['status']}")
    
    # Check if the solver was successful
    if result['q_ddot'] is not None:
        print(f"Objective value: {result['objective_value']:.4f}")
        print("\nJoint accelerations (q_ddot):")
        print(result['q_ddot'])
        print("\nContact forces (u):")
        print(result['u'])
        print("\nJoint torques (tau):")
        print(result['tau'])
        
        # Visualize the results
        visualize_results(result, q, q_dot)
    else:
        print("Solver failed to find a solution.")
    
    return result

def visualize_results(result, q, q_dot):
    """Visualize the optimization results"""
    q_ddot = result['q_ddot']
    u = result['u']
    tau = result['tau']
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Plot joint accelerations
    ax1 = fig.add_subplot(311)
    ax1.bar(range(len(q_ddot)), q_ddot)
    ax1.set_title('Joint Accelerations')
    ax1.set_xlabel('Joint Index')
    ax1.set_ylabel('Acceleration (rad/s²)')
    ax1.grid(True)
    
    # Plot contact forces
    ax2 = fig.add_subplot(312)
    ax2.bar(range(len(u)), u)
    ax2.set_title('Contact Forces')
    ax2.set_xlabel('Force Component Index')
    ax2.set_ylabel('Force (N) / Torque (Nm)')
    ax2.grid(True)
    
    # Plot joint torques
    ax3 = fig.add_subplot(313)
    ax3.bar(range(len(tau)), tau)
    ax3.set_title('Joint Torques')
    ax3.set_xlabel('Joint Index')
    ax3.set_ylabel('Torque (Nm)')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Optionally, simulate and animate the motion
    simulate_motion(q, q_dot, q_ddot)

def simulate_motion(q_init, q_dot_init, q_ddot):
    """Simulate robot motion using the computed accelerations"""
    # Simulation parameters
    dt = 0.01  # Time step
    sim_time = 1.0  # Total simulation time
    n_steps = int(sim_time / dt)
    
    # Initialize state trajectories
    q_traj = np.zeros((n_steps, len(q_init)))
    q_dot_traj = np.zeros((n_steps, len(q_dot_init)))
    
    # Set initial conditions
    q_traj[0] = q_init
    q_dot_traj[0] = q_dot_init
    
    # Simple numerical integration (Euler method)
    for i in range(1, n_steps):
        # Update velocities using accelerations
        q_dot_traj[i] = q_dot_traj[i-1] + q_ddot * dt
        
        # Update positions using velocities
        q_traj[i] = q_traj[i-1] + q_dot_traj[i] * dt
    
    # For simplicity, we'll just plot the joint trajectories
    plt.figure(figsize=(10, 6))
    
    for j in range(min(6, len(q_init))):  # Plot first 6 joints for clarity
        plt.plot(np.linspace(0, sim_time, n_steps), q_traj[:, j], 
                 label=f'Joint {j+1}')
    
    plt.title('Joint Position Trajectories (First 6 Joints)')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Position (rad)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_wbc_example()