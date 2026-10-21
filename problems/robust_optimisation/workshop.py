# %% [markdown]
# # Robust Optimisation via Sum of Largest Eigenvalues
#
# ## Problem description
#
# For a symmetric matrix $M \in \mathbb{S}^n$ with eigenvalues $\lambda_1(M) \geq \lambda_2(M) \geq \dots \geq \lambda_n(M)$, the sum of the $k$ largest eigenvalues is
#
# $f_k(M) = \sum_{i=1}^k \lambda_i(M).$
#
# This function appears whenever only the dominant modes of a system matter, for example the variance captured by the top $k$ principal components of a covariance matrix, or the $k$ least stable modes of a dynamical system. Although eigenvalues are not convex functions of $M$ individually, $f_k$ is convex (Ky Fan, 1949).
#
# A system is described by a symmetric matrix that depends on $p$ design parameters $x \in \mathbb{R}^p$:
#
# $A(x) = A_0 + \sum_{i=1}^p x_i A_i, \quad -1 \leq x_i \leq 1.$
#
# The designer wishes to choose $x$ to minimise $f_k(A(x))$.
#
# In practice the effect of each parameter is not known exactly. Suppose that $A_i$ is really $A_i + \delta_i E_i$ for some unknown $\delta_i \in [-\rho, \rho]$, so that
#
# $A(x, \delta) = A_0 + \sum_{i=1}^p x_i (A_i + \delta_i E_i).$
#
# A **robust** design minimises the worst case of $f_k(A(x, \delta))$ over all possible $\delta$. This workshop solves both the nominal and the robust problem and compares the resulting designs.
#
# ## Problem formulation
#
# ### SDP representation of $f_k$
#
# The key result is that $f_k$ can be written as the optimal value of a semidefinite program:
#
# $f_k(M) = \min_{t \in \mathbb{R}, \, S \in \mathbb{S}^n} \left\{ kt + \text{tr}(S) : S \succeq 0, \; S \succeq M - tI \right\}.$
#
# **Exercise:** show that this holds. Hint: fix $t$ first and bound $\text{tr}(S)$ from below using the eigenvectors of $M$, then minimise over $t$.
#
# ### Nominal problem
#
# The model variables are the following:
# * $x \in \mathbb{R}^p$: Design parameters.
# * $t \in \mathbb{R}$, $S \in \mathbb{S}^n$: Auxiliary variables from the SDP representation of $f_k$.
#
# The objective is:
#
# $\min kt + \text{tr}(S)$,
#
# subject to:
#
# $S \succeq 0,$
#
# $S \succeq A(x) - tI,$
#
# $-1 \leq x_i \leq 1, \quad \forall i \in \{1, \dots, p\}.$
#
# Since $A(x)$ is affine in $x$, the constraint $S \succeq A(x) - tI$ is a linear matrix inequality and the whole problem is an SDP.
#
# ### Robust problem
#
# For fixed $x$, the map $\delta \mapsto f_k(A(x, \delta))$ is a convex function composed with an affine map, so it is convex. The maximum of a convex function over the box $[-\rho, \rho]^p$ is attained at one of its $2^p$ vertices $\mathcal{V} = \{-\rho, \rho\}^p$. The robust problem is therefore equivalent to
#
# $\min_x \max_{v \in \mathcal{V}} f_k(A(x, v)),$
#
# which, using a separate copy of the SDP representation for each vertex, becomes:
#
# $\min z$,
#
# subject to:
#
# $k t_v + \text{tr}(S_v) \leq z, \quad \forall v \in \mathcal{V},$
#
# $S_v \succeq 0, \quad \forall v \in \mathcal{V},$
#
# $S_v \succeq A(x, v) - t_v I, \quad \forall v \in \mathcal{V},$
#
# $-1 \leq x_i \leq 1, \quad \forall i \in \{1, \dots, p\}.$
#
# ## Required packages

# %%
import itertools
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# %% [markdown]
# ## Data generation
#
# The following code generates a random base matrix $A_0$, design directions $A_1, \dots, A_p$ and uncertainty directions $E_1, \dots, E_p$, all symmetric. The values of `n`, `p`, `k` and `rho` can be changed.

# %%
n = 8 # Matrix dimension
p = 4 # Number of design parameters
k = 3 # Number of largest eigenvalues to sum
rho = 0.8 # Uncertainty level, delta_i in [-rho, rho]

rng = np.random.default_rng(2026)

def random_symmetric(n, rng):
    B = rng.standard_normal((n, n))
    return (B + B.T) / 2

A0 = random_symmetric(n, rng) + 2 * np.eye(n)
A_list = [random_symmetric(n, rng) for i in range(p)]
E_list = [random_symmetric(n, rng) for i in range(p)]

# %% [markdown]
# The following helper functions evaluate $A(x, \delta)$ and $f_k$ numerically, and list the vertices of the uncertainty box.

# %%
def A_of(x, delta=None):
    """Return A(x, delta) as a numpy array. delta=None gives the nominal matrix A(x)."""
    if delta is None:
        delta = np.zeros(p)
    return A0 + sum(x[i] * (A_list[i] + delta[i] * E_list[i]) for i in range(p))

def sum_top_k_eigs(M, k):
    """Sum of the k largest eigenvalues of a symmetric matrix."""
    eigs = np.linalg.eigvalsh(M) # Sorted ascending
    return np.sum(eigs[-k:])

# All 2^p vertices of the box [-rho, rho]^p
vertices = [np.array(v) for v in itertools.product([-rho, rho], repeat=p)]

print(f"Sum of top-{k} eigenvalues of A0 (doing nothing, x = 0): {sum_top_k_eigs(A0, k):.4f}")
print(f"Number of vertices of the uncertainty box: {len(vertices)}")

# %% [markdown]
# ## Checking the SDP representation
#
# Before using it inside a larger model, the following code checks that the SDP representation of $f_k$ agrees with an eigenvalue computation on $A_0$.

# %%
# Write code here


# %% [markdown]
# Notice that the optimal $t$ lies between the $(k+1)$-th and $k$-th largest eigenvalues, as the derivation predicts. Which point in this interval the solver returns depends on the algorithm.
#
# ## Model building and solving (nominal)
#
# The following code builds and solves the nominal problem using CVXPY.

# %%
def solve_nominal(A0, A_list, k):
    """
    Solve the nominal problem.
    Returns the optimal design x (numpy array of length p) and the optimal value.
    """

    # Write code here
    x_opt = None
    value = None

    return x_opt, value

x_nominal, value_nominal = solve_nominal(A0, A_list, k)
if x_nominal is not None:
    print(f"Nominal design x: {np.round(x_nominal, 3)}")
    print(f"Nominal objective: {value_nominal:.4f}")

# %% [markdown]
# ## Solution handling (nominal)
#
# The following code plots the eigenvalues of $A_0$ and of the optimised matrix $A(x)$. The shaded region marks the $k$ largest eigenvalues, whose sum is being minimised.

# %%
def plot_spectra(matrices, labels, k):
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(matrices)
    for j, (M, label) in enumerate(zip(matrices, labels)):
        eigs = np.sort(np.linalg.eigvalsh(M))[::-1]
        ax.bar(np.arange(1, len(eigs) + 1) + (j - (len(matrices) - 1) / 2) * width, eigs, width, label=label)
    ax.axvspan(0.5, k + 0.5, color='grey', alpha=0.15, label=f'Top {k}')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Eigenvalue index (descending)')
    ax.set_ylabel('Eigenvalue')
    ax.legend()
    plt.show()

if x_nominal is not None:
    plot_spectra([A0, A_of(x_nominal)], [r'$A_0$', r'$A(x_{nom})$'], k)
else:
    print("Complete solve_nominal above to see the comparison.")

# %% [markdown]
# Compare the two spectra. Did every one of the top $k$ eigenvalues decrease? What has happened to the remaining eigenvalues, and why does the optimiser not care about them?
#
# ## Why additive uncertainty is not interesting
#
# A natural first attempt at a robust model is $A(x) + \Delta$ with $\|\Delta\|_2 \leq \epsilon$ in spectral norm. In that case $f_k(A(x) + \Delta) \leq f_k(A(x)) + k\epsilon$, with equality for $\Delta^\ast = \epsilon V_k V_k^T$ where $V_k$ holds the top $k$ eigenvectors of $A(x)$. The worst case is the nominal value plus a constant, so the robust design is identical to the nominal one.
#
# The following code checks this for $A_0$.

# %%
epsilon = 0.3

eigs, V = np.linalg.eigh(A0) # Ascending order
V_k = V[:, -k:] # Top-k eigenvectors
Delta_star = epsilon * V_k @ V_k.T # Spectral norm equals epsilon

print(f"Worst-case value: {sum_top_k_eigs(A0 + Delta_star, k):.4f}")
print(f"f_k(A0) + k*eps:  {sum_top_k_eigs(A0, k) + k*epsilon:.4f}")

# %% [markdown]
# For robustness to change the design, the uncertainty has to interact with the decision. That is why the model above lets the uncertainty enter through the coefficients $A_i + \delta_i E_i$ that multiply $x_i$.
#
# ## Model building and solving (robust)
#
# The following code builds and solves the robust problem, with one SDP block per vertex of the uncertainty box.

# %%
def solve_robust(A0, A_list, E_list, k, vertices):
    """
    Solve the robust problem, using one copy of the SDP representation per vertex.
    Returns the optimal design x (numpy array of length p) and the optimal worst-case value.
    """

    # Write code here
    x_opt = None
    value = None

    return x_opt, value

x_robust, value_robust = solve_robust(A0, A_list, E_list, k, vertices)
if x_robust is not None:
    print(f"Robust design x: {np.round(x_robust, 3)}")
    print(f"Robust worst-case objective: {value_robust:.4f}")

# %% [markdown]
# The size of this model grows with $2^p$. What would happen with $p = 20$ design parameters?
#
# ## Solution handling (robust)
#
# The following code compares the two designs on three measures: their value with no uncertainty, their worst case over the vertices, and their behaviour on uncertainty sampled uniformly from the box.

# %%
def worst_case(x):
    return max(sum_top_k_eigs(A_of(x, v), k) for v in vertices)

# Only compare the designs that have been solved
designs = {name: x for name, x in {'Nominal': x_nominal, 'Robust': x_robust}.items() if x is not None}

print(f"{'Design':<20}{'No uncertainty':>16}{'Worst case':>12}")
for name, x in {'Do nothing (x = 0)': np.zeros(p), **designs}.items():
    print(f"{name:<20}{sum_top_k_eigs(A_of(x), k):>16.4f}{worst_case(x):>12.4f}")

# %% [markdown]
# The following code samples 5000 uncertainty realisations and plots the distribution of $f_k(A(x, \delta))$ for each design. The dashed lines mark each design's worst case over the vertices, and the solid black line marks doing nothing, which is unaffected by $\delta$.

# %%
def plot_scenarios(designs, n_samples=5000, seed=2026):
    rng = np.random.default_rng(seed)
    samples = rng.uniform(-rho, rho, size=(n_samples, p))

    values = {name: [sum_top_k_eigs(A_of(x, d), k) for d in samples] for name, x in designs.items()}

    # Shared bins so the distributions are directly comparable
    all_values = [sum_top_k_eigs(A0, k)] + [v for vals in values.values() for v in vals]
    bins = np.histogram_bin_edges(all_values, bins=60)

    fig, ax = plt.subplots(figsize=(10, 4))
    for (name, x), colour in zip(designs.items(), ['tab:blue', 'tab:orange']):
        ax.hist(values[name], bins=bins, alpha=0.5, color=colour, label=name)
        ax.axvline(worst_case(x), color=colour, linestyle='--', label=f'{name} worst case')
    ax.axvline(sum_top_k_eigs(A0, k), color='black', label='Do nothing (x = 0)')
    ax.set_xlabel(r'$f_k(A(x, \delta))$')
    ax.set_ylabel('Number of samples')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plt.show()

plot_scenarios(designs)

# %% [markdown]
# Some questions to consider:
# * How does the worst case of the nominal design compare with doing nothing at all?
# * How much does the robust design give up when there is no uncertainty? This is sometimes called the *price of robustness*.
# * Why does no sampled value exceed the dashed line for its design?
# * Re-run with `rho = 0` and with larger values. How do the two designs change?
