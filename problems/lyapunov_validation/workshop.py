# %% [markdown]
# # Lyapunov function validation
#
# ## Definition
#
# A [Lyapunov function](https://en.wikipedia.org/wiki/Lyapunov_function) for an autonomous dynamical system described by
#
# $\begin{cases} 
# g: \mathbb{R}^n \rightarrow \mathbb{R}^n, \\
# \dot y = g(y) \end{cases}$
#
# with equilibrium point at $y = 0$ is a scalar function $V:\mathbb{R}^n \rightarrow \mathbb{R}$ that is:
# * Continuous
# * Has continuous first derivatives
# * Is strictly positive for $y \neq 0$
# * Time derivative $\dot V = \Delta V \cdot g$ is non-positive.
#
# ## Required packages

# %%
import cvxpy as cp
import numpy as np
import cvxopt # needed as a solver backend


# %% [markdown]
# ## Polynomial example
#
# Consider the dynamical system described by $\dot x = -x$. 
#
# Use SDP to verify the 4th condition for $V(x) = x^2$ is a valid Lyapunov function for this system.
#
# (Hint 1: Consider the form $f(t) = p_i t^i$)
#
# (Hint 2: [Theorem 11.2](https://www.damtp.cam.ac.uk/user/hf323/lent2017_topics_convex_optimisation/lecture11.pdf))

# %%

# %% [markdown]
# ## Function for checking
#
# If you have not already, construct a function that takes a tuple, $P$, representing a polynomial and checks if it is non-negative.
#
# (Hint 3: Name the function `check_poly_nonnegative(P)` and ensure it returns boolean values, `True` if non-negative and `False` if not)

# %%
def check_poly_nonnegative(P: tuple) -> bool:
    
    # actual code here
    
    return True


# %% [markdown]
# ## Polynomials to check
#
# The following $P$ vectors represent different polynomials in different sized spaces. Check if your function checks them correctly.

# %%
polynomials = {
    (0,0,1): True, # x^2
    (0,0,-1): False, # -x^2
    (0,0,0): True, # 0
    (0,1): False, # x
    (1,1,1): True, # 1 + x + x^2
    (0,1,0): False, # x
    (0,1,1,0): False, # x + x^2
    (0,1,-1,0): False, # x + -x^2
}

for i, P in enumerate(polynomials):
    print(f"Polynomial check {i+1} passed: {check_poly_nonnegative(P) == polynomials[P]}")

# %% [markdown]
# ## More complicated dynamical system
#
# Consider the dynamical system
#
# $\begin{cases} 
# \dot x_1: -x_1 -x_2^2, \\
# \dot x_2: -x_2 \end{cases}$
#
# A suitable Lyapunov candidate is the following:
#
# $V(x_1, x_2) = \frac{1}{2} x_1^2 + \frac{1}{2} x_2^2$
#
# Prove that this is a valid Lyapunov function using your function `check_poly_nonnegative(P)`

# %%
