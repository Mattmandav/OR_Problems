# %% [markdown]
# # Robust Optimization via Sum of Largest Eigenvalues
# 
# Two problems:
#   1. **Robust Design (SDP)**: Minimize the sum of the top-k eigenvalues of a matrix A subject to convex constraints.
# 
#   2. **Worst-case analysis**: Find the worst-case sum of top-$k$ eigenvalues of a covariance matrix under spectral-norm bounded perturbations.
# 
# ## Required packages

# %%
import random
import numpy as np
import cvxpy as cp
import cvxopt

# %% [markdown]
# ## Data generation

# %%
random.seed(2026)

n = 6 # matrix dimension
k = 2 # number of top eigenvalues to sum
epsilon = 0.3 # perturbation budget (spectral norm)

# Generate a random PSD base covariance matrix Sigma_0
A = np.random.randn(n, n)
Sigma_0 = A @ A.T / n + 0.5 * np.eye(n)   # guaranteed PSD

# Sum of top-k eigenvalues
def sum_top_k_eigs(M, k):
    eigs = np.linalg.eigvalsh(M) # sorted ascending
    return np.sum(eigs[-k:])

# %% [markdown]
# ## Robust design (SDP)
# 
# Design a symmetric PSD matrix $A$ that minimizes the sum of its top-$k$ eigenvalues subject to:
# 
# $tr(A) = 1 \quad$ (normalisation — energy constraint)
#   
# $A  \succeq  0$
#   
# $\text{diag}(A) \geq 0.02 \quad$ (each diagonal entry bounded below)
# 
# ### Reformulation:
# 
# $\min_{A, t, S} [kt + tr(S)]$
# 
# subject to:
#   
# $A - tI \preceq S$
#     
# $S  \succeq  0$
#     
# $tr(A) = 1$
#     
# $A  \succeq  0$
# 
# $\text{diag}(A) \geq 0.02$
# 
# ### Interpretation
#
# Find the "most spread-out" PSD matrix on the trace-simplex (minimise concentration of variance in any $k$ directions).

# %%


# %% [markdown]
# ## Worst-case analysis
# 
# Maximize  sum of top-$k$ eigenvalues of ($\Sigma_0 + \Delta$) subject to:
# 
# $||\Delta||_2 \leq \epsilon \quad$(spectral norm)
# 
# $\Sigma_0 + \Delta \succeq 0$
# 
# ### Closed-form worst-case perturbation
# 
# For the spectral norm ball, the worst $\Delta$ is:
# 
# $\Delta^\ast = \epsilon V_k V_k^T$
# 
# where $V_k$ are the top-$k$ eigenvectors of $\Sigma_0$.
# 
# ### Interpretation
# 
# This pushes each of the top-$k$ eigenvalues up by exactly epsilon, and does not affect the remaining eigenvalues.
# 
# The worst-case value is therefore:  `sum_k_lambda`$(\Sigma_0) + \epsilon k$

# %%



