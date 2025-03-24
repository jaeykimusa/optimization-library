% Quadratic program with linear constraints

% Author: jaeykimusa

clear; clc; close all;

%% Symbolic Computation (Using Symbolic Math Toolbox)

syms x1 x2
f = 0.5*x1^2 + x2^2 - x1*x2 - 2*x1 - 6*x2;

% Compute Hessian (H) and linear term (f)
H = hessian(f, [x1, x2]);            % Hessian matrix
grad_f = gradient(f, [x1, x2]);      % Gradient
f_vec = subs(grad_f, [x1, x2], [0, 0]); % Extract linear coefficients at x=0

% Convert symbolic matrices to numeric
H_numeric = double(H);
f_numeric = double(f_vec);

% Constraints
A = [1, 1; -1, 2; 2, 1];
b = [2; 2; 3];

% Solve using quadprog
[x_opt, f_val] = quadprog(H_numeric, f_numeric, A, b);

% Print results
fprintf('Optimal x1: %.4f\n', x_opt(1));
fprintf('Optimal x2: %.4f\n', x_opt(2));
fprintf('Minimum value of f(x): %.4f\n', f_val);

%% Problem-Based Optimization

% Define optimization variables
x = optimvar('x', 2);

% Define objective function
obj = 0.5*x(1)^2 + x(2)^2 - x(1)*x(2) - 2*x(1) - 6*x(2);

% Set up problem
prob = optimproblem('Objective', obj);

% Add constraints
prob.Constraints.c1 = x(1) + x(2) <= 2;
prob.Constraints.c2 = -x(1) + 2*x(2) <= 2;
prob.Constraints.c3 = 2*x(1) + x(2) <= 3;

% Solve
[sol, f_val] = solve(prob);
x_opt = sol.x;

% Print results
fprintf('Optimal x1: %.4f\n', x_opt(1));
fprintf('Optimal x2: %.4f\n', x_opt(2));
fprintf('Minimum value of f(x): %.4f\n', f_val);

%% General Method

% objective function: convert to the form 1/2x^THx+f^Tx
H = [1, -1; -1, 2];
f = [-2; -6];

% constraints: express as Ax <= b.
A = [1, 1; -1, 2; 2, 1];
b = [2; 2; 3];

% qp:
[x, fval] = quadprog(H, f, A, b);

% Print results
fprintf('Optimal x1: %.4f\n', x(1));
fprintf('Optimal x2: %.4f\n', x(2));
fprintf('Minimum value of f(x): %.4f\n', fval);
