import mpi4py.MPI as MPI
import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize

EPS = 1e-15 
#---------------------Initialize MPI---------------------#
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#---------------------Data read and distribution---------------------#
BASE_DIR = "/gpfs/projects/AMS598/projects2025_data/project3_data"
FILE_NAME = f"project3_data_part{rank}.csv"

try:

    file_path = os.path.join(BASE_DIR, FILE_NAME)
    data_i = pd.read_csv(file_path)

    y_i = data_i['y'].values
    X_i = data_i.drop(columns=['y']).values

    n_i, p_minus_1 = X_i.shape

    intercept_col = np.ones((n_i, 1))
    X_i = np.hstack((intercept_col, X_i))

    p = X_i.shape[1]

    if rank == 0:
        print("--------------------------------------------------")
        print(f"ADMM Logistic Regression Initialization (N={size} Processes)")
        print("--------------------------------------------------")
    
    print(f"Process {rank:02d}/{size}: Loaded data from {FILE_NAME}. Samples: {n_i}, Features: {p}")
    
except Exception as e:

    if rank == 0:
        print(f"ERROR: Could not load data. Check base directory and file format. Error: {e}")
    comm.Abort(1)

#---------------------Phase-2-----------------------------
def sigmoid(z):
    """Sigmoid function: 1 / (1 + exp(-z))."""
    z = np.clip(z, -500, 500) 
    return 1.0 / (1.0 + np.exp(-z))

def compute_local_objective(beta_i, X_i, y_i, beta_bar_k, u_i_k, rho):
    """
    Computes the value of the local Augmented Lagrangian Objective function.
    
    f(beta_i) = l_i(beta_i) + (rho/2) * ||beta_i - beta_bar_k + u_i_k||^2
    
    Arguments:
        beta_i (np.array): The local coefficient vector being minimized (the variable).
        X_i, y_i (np.array): Local data.
        beta_bar_k (np.array): Consensus vector from the last iteration (fixed parameter).
        u_i_k (np.array): Local dual variable from the last iteration (fixed parameter).
        rho (float): The ADMM penalty parameter (fixed parameter).
        
    Returns:
        float: The scalar value of the objective function.
    """
    z = X_i @ beta_i
    p_i = sigmoid(z)
    p_i = np.clip(p_i, EPS, 1 - EPS)
    
    log_likelihood_cost = -np.sum(y_i * np.log(p_i) + (1.0 - y_i) * np.log(1.0 - p_i))
    
    admm_term = (rho / 2.0) * np.sum((beta_i - (beta_bar_k - u_i_k))**2)

    return log_likelihood_cost + admm_term


def compute_local_gradient(beta_i, X_i, y_i, beta_bar_k, u_i_k, rho):
    """
    Computes the gradient of the local Augmented Lagrangian Objective function.
    
    Providing the gradient (Jacobian) greatly speeds up the optimizer.
    
    Returns:
        np.array: The vector of the gradient.
    """
    
    z = X_i @ beta_i
    p_i = sigmoid(z)
    
    grad_log_likelihood = X_i.T @ (p_i - y_i)
    
    grad_admm_term = rho * (beta_i - (beta_bar_k - u_i_k))

    return grad_log_likelihood + grad_admm_term

# ----------------------------------------------------------------------Phase 3----------------

beta_i = np.zeros(p)
beta_bar = np.zeros(p)
u_i = np.zeros(p)
rho = 1.0 
MAX_ITERATIONS = 100 

if rank == 0:
    print(f"Phase 2 Complete: Local objective and gradient functions are defined.")
    print(f"Starting ADMM iterations (Phase 3)...")

rho = 1.0 
MAX_ITERATIONS = 100

beta_i = np.zeros(p)  
beta_bar = np.zeros(p) 
u_i = np.zeros(p)      


if rank == 0:
    beta_bar_history = [beta_bar.copy()]
    print(f"ADMM process starting for {MAX_ITERATIONS} iterations...")

for k in range(MAX_ITERATIONS):

    local_args = (X_i, y_i, beta_bar, u_i, rho)

    res = minimize(
        fun=compute_local_objective, 
        x0=beta_i, 
        args=local_args,
        method='L-BFGS-B', 
        jac=compute_local_gradient,
        options={'disp': False, 'maxiter': 50}
    )

    beta_i_kplus1 = res.x

    sum_betas = np.zeros(p)
    comm.Allreduce(beta_i_kplus1, sum_betas, op=MPI.SUM)
    beta_bar_kplus1 = sum_betas / size
    beta_bar = beta_bar_kplus1 

    u_i = u_i + (beta_i_kplus1 - beta_bar_kplus1)
    
    beta_i = beta_i_kplus1
    
    if rank == 0:
        beta_bar_history.append(beta_bar.copy())
        
        if (k + 1) % 10 == 0:
            print(f"Iteration {k+1:03d}: ADMM cycle complete. Consensus vector updated.")


# ----------------------------------------------------------------------
# PHASE 4: FINALIZATION & OUTPUT
# ----------------------------------------------------------------------

comm.Barrier()

if rank == 0:
    print("\n--------------------------------------------------")
    print("PHASE 4: ADMM CONVERGENCE COMPLETE")
    print(f"Final Consensus Estimates (Coefficients, p={p}):")
    print(np.array2string(beta_bar, precision=6, separator=', ', suppress_small=True))
    print("--------------------------------------------------")