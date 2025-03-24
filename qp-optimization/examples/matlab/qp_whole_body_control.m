function [q_ddot, u] = solve_wbc_qp(H, h, J_c, Jc_dot_q_dot, tasks, w_f, u_d, mu, f_min, S_inv, tau_m, tau_M)
    % Define optimization variables: x = [q_ddot; u]
    nq = 18; % Number of joints (q_ddot is 18x1)
    nu = 12; % Contact forces (u is 12x1)
    nx = nq + nu;
    
    % Construct the objective function Hessian and linear term
    H_obj = zeros(nx);
    f_obj = zeros(nx, 1);
    
    % Task terms
    for i = 1:length(tasks)
        J_i = tasks(i).J; % Task Jacobian (e.g., 6x18)
        Jdot_q_dot_i = tasks(i).Jdot_q_dot; % J̇_i q̇ (6x1)
        v_d_i = tasks(i).v_d; % Desired task acceleration (6x1)
        w_i = tasks(i).w; % Task weight (scalar)
        
        % Weight matrix for the task
        W_i = (w_i^2) * eye(size(J_i,1));
        
        % Hessian contribution (2 * J_i' * W_i * J_i)
        H_obj(1:nq, 1:nq) = H_obj(1:nq, 1:nq) + 2 * J_i' * W_i * J_i;
        
        % Linear term contribution (2 * J_i' * W_i * (Jdot_q_dot_i - v_d_i))
        f_obj(1:nq) = f_obj(1:nq) + 2 * J_i' * W_i * (Jdot_q_dot_i - v_d_i);
    end
    
    % Force term (u - u_d)
    W_f = 2 * (w_f^2) * eye(nu);
    H_obj(nq+1:end, nq+1:end) = W_f;
    f_obj(nq+1:end) = -2 * (w_f^2) * u_d;
    
    % Equality constraints (dynamics and contact acceleration)
    A_eq = [H, -J_c';
            J_c, zeros(size(J_c,1), nu)];
    b_eq = [-h;  % H q̈ - J_c' u = -h
            -Jc_dot_q_dot];  % J_c q̈ = -J̇_c q̇
    
    % Inequality constraints (friction cone and torque limits)
    % Friction cone for each foot (4 feet)
    A_ineq_friction = [];
    b_ineq_friction = [];
    for j = 1:4
        idx = (3*(j-1)+1) : (3*j); % Indices for foot j's forces
        A_foot = [1, 0, -mu;
                 -1, 0, -mu;
                  0, 1, -mu;
                  0, -1, -mu;
                  0, 0, -1]; % f_z >= f_min
        b_foot = [0; 0; 0; 0; -f_min];
        
        % Place in full matrix
        A_foot_full = zeros(5, nx);
        A_foot_full(:, nq+idx) = A_foot;
        A_ineq_friction = [A_ineq_friction; A_foot_full];
        b_ineq_friction = [b_ineq_friction; b_foot];
    end
    
    % Torque constraints: τ_m ≤ S⁻¹ (H q̈ + h - J_c' u) ≤ τ_M
    A_tau1 = S_inv * H; % S⁻¹ * H (12x18)
    A_tau2 = -S_inv * J_c'; % S⁻¹ * (-J_c') (12x12)
    A_tau_row1 = [A_tau1, A_tau2]; % Upper bound
    A_tau_row2 = [-A_tau1, -A_tau2]; % Lower bound
    A_ineq_tau = [A_tau_row1; A_tau_row2];
    
    b_tau1 = tau_M - S_inv * h; % Upper bound
    b_tau2 = -tau_m + S_inv * h; % Lower bound
    b_ineq_tau = [b_tau1; b_tau2];
    
    % Combine all inequalities
    A_ineq = [A_ineq_friction; A_ineq_tau];
    b_ineq = [b_ineq_friction; b_ineq_tau];
    
    % Solve QP using quadprog
    options = optimoptions('quadprog', 'Display', 'off');
    x = quadprog(H_obj, f_obj, A_ineq, b_ineq, A_eq, b_eq, [], [], [], options);
    
    % Extract results
    q_ddot = x(1:nq);
    u = x(nq+1:end);
end