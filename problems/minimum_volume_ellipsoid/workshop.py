# %% [markdown]
# # Minimum Volume Enclosing Ellipsoid (Löwner-John Ellipsoid)
#
# ## Problem description
#
# Given a set of points $x_1, \dots, x_N \in \mathbb{R}^n$, find the **smallest volume ellipsoid** that contains all points.
#
# An ellipsoid centred at $c \in \mathbb{R}^n$ can be represented as:
#
# $\mathcal{E} = \{ x \in \mathbb{R}^n : (x - c)^T P^{-1} (x - c) \leq 1 \}$
#
# where $P \in \mathbb{S}^n_{++}$ is a **symmetric positive definite** matrix that determines the shape and orientation of the ellipsoid. 
#
# The semi-axes of $\mathcal{E}$ are the eigenvectors of $P$, with lengths given by the square roots of the corresponding eigenvalues.
#
# ## Volume of an Ellipsoid
#
# The volume of $\mathcal{E}$ is proportional to:
#
# $\text{Vol}(\mathcal{E}) \propto \sqrt{\det P} = \exp\left(\frac{1}{2} \log \det P\right)$
#
# Minimising the volume is therefore equivalent to minimising $\log \det P$, or equivalently **maximising** $\log \det P^{-1}$.
#
# ## Required packages

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import cvxpy as cp
import gurobipy as gp
from scipy.optimize import minimize


# %% [markdown]
# The following code generates a number of points based on one of several ''modes''.

# %%
def generate_points(mode='clusters', n_points=30, seed=2026):
    """
    Generate interesting 2D point sets for the mve problem.

    Modes
    -----
    'clusters'    : several Gaussian clusters
    'annulus'     : points on a ring
    'simplex'     : points near vertices of a triangle plus interior noise
    'ellipse'     : points near the boundary of a tilted ellipse
    'cross'       : points arranged in a cross shape
    """
    rng = np.random.default_rng(seed)

    if mode == 'clusters':
        centres = np.array([[0, 0], [3, 1], [1, 3], [-1, 2]])
        points = np.vstack([
            rng.multivariate_normal(c, 0.2 * np.eye(2), n_points // len(centres))
            for c in centres
        ])

    elif mode == 'annulus':
        angles = rng.uniform(0, 2 * np.pi, n_points)
        radii  = rng.uniform(1.5, 2.0, n_points)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    elif mode == 'simplex':
        vertices = np.array([[0, 0], [4, 0], [2, 3.5]])
        # Points near each vertex plus some interior scatter
        boundary = np.vstack([
            v + rng.normal(0, 0.15, (n_points // 3, 2)) for v in vertices
        ])
        interior = np.column_stack([
            rng.uniform(0.5, 3.5, n_points // 4),
            rng.uniform(0.2, 2.5, n_points // 4)
        ])
        points = np.vstack([boundary, interior])

    elif mode == 'ellipse':
        angles = rng.uniform(0, 2 * np.pi, n_points)
        # Tilted ellipse with semi-axes 3 and 1, rotated 45 degrees
        theta  = np.pi / 4
        R      = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])
        points = np.column_stack([3 * np.cos(angles), np.sin(angles)]) @ R.T
        points += rng.normal(0, 0.1, points.shape)

    elif mode == 'cross':
        horizontal = np.column_stack([
            rng.uniform(-3, 3, n_points // 2),
            rng.normal(0, 0.2, n_points // 2)
        ])
        vertical = np.column_stack([
            rng.normal(0, 0.2, n_points // 2),
            rng.uniform(-3, 3, n_points // 2)
        ])
        points = np.vstack([horizontal, vertical])

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return points


# %% [markdown]
# The following function plots the set of 2D points and optionally an ellipsoid found by the solver.

# %%
def plot_mve(points, P=None, c=None, title=None, ax=None):
    """
    Plot 2D points and optionally the enclosing ellipsoid.

    Parameters
    ----------
    points : np.ndarray (N, 2)
    P      : np.ndarray (2, 2) - ellipsoid shape matrix (optional)
    c      : np.ndarray (2,)   - ellipsoid centre (optional)
    title  : str (optional)
    ax     : matplotlib axis (optional)
    """
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Points
    ax.scatter(points[:, 0], points[:, 1],
               color='steelblue', s=30, zorder=5, label='Data points')

    # Ellipsoid
    if P is not None and c is not None:
        # Eigendecomposition of P gives axes and orientation
        eigvals, eigvecs = np.linalg.eigh(P)
        # Semi-axis lengths are sqrt of eigenvalues
        width, height = 2 * np.sqrt(eigvals)
        # Rotation angle from eigenvectors
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

        ellipse = Ellipse(
            xy=c, width=width, height=height, angle=angle,
            edgecolor='crimson', facecolor='crimson',
            linewidth=2, alpha=0.15, zorder=3,
            label='Min. volume ellipsoid'
        )
        ax.add_patch(ellipse)

        # Plot ellipse boundary
        ellipse_edge = Ellipse(
            xy=c, width=width, height=height, angle=angle,
            edgecolor='crimson', facecolor='none',
            linewidth=2, zorder=4
        )
        ax.add_patch(ellipse_edge)

        # Mark centre
        ax.plot(*c, '+', color='crimson', markersize=12,
                markeredgewidth=2, zorder=6, label='Centre')

        # Highlight support points (those on the boundary)
        diffs   = points - c
        Q       = np.linalg.inv(P)
        vals    = np.einsum('ij,jk,ik->i', diffs, Q, diffs)
        support = np.where(np.abs(vals - 1) < 0.05)[0]
        if len(support) > 0:
            ax.scatter(points[support, 0], points[support, 1],
                       s=80, facecolors='none', edgecolors='navy',
                       linewidths=1.5, zorder=7, label='Support points')

    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.set_title(title or 'Minimum Volume Enclosing Ellipsoid', fontsize=11)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.tight_layout()

    if show:
        plt.show()

    return ax


# %%
modes = ['clusters', 'annulus', 'simplex', 'ellipse', 'cross']
n = 40
s = 2026

# Setting up plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Generating data
for i, mode in enumerate(modes):
    points = generate_points(mode=mode, n_points=n, seed=s)
    plot_mve(points, title=f'Mode: {mode}', ax=axes[i])

# Share common plot legend
axes[-1].axis('off')
handles, labels = axes[0].get_legend_handles_labels()
for ax in axes[:-1]:
    ax.get_legend().remove()
axes[-1].legend(handles, labels, fontsize=11, loc='center',
                title='Legend', title_fontsize=12,
                frameon=False)

# Finalise plot
plt.suptitle('Minimum Volume Enclosing Ellipsoid solved using CVXPY (SDP)', fontsize=13)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Standard Formulation
#
# $\min_{P,\ c} \quad \log \det P^{-1}$
#
# subject to
#
# $(x_i - c)^T P^{-1} (x_i - c) \leq 1, \quad i = 1, \dots, N$
#
# $P \succ 0$
#
# This is a **convex optimisation problem**: the objective $\log \det P^{-1}$ is convex in $P^{-1}$, and the constraints are convex in $(P, c)$. However, the decision variables appear nonlinearly, so we apply a change of variables to obtain a standard SDP.
#
# ## SDP Formulation
#
# $\max_{B,\ d} \quad \log \det B$
#
# subject to
#
# $\begin{pmatrix} 1 & (Bx_i + d)^T \\ Bx_i + d & I \end{pmatrix} \succeq 0, \quad i = 1, \dots, N$
#
# $B \succeq 0$
#
# The ellipsoid centre and matrix are then recovered as:
#
# $c = -B^{-1} d, \qquad P = B^{-T} B^{-1} = (B^T B)^{-1}$

# %%
def solve_mve_cvxpy(points):
    """
    Solve the mve problem in the SDP formulation using CVXPY.

    Returns
    -------
    P : np.ndarray (n, n) - ellipsoid shape matrix
    c : np.ndarray (n,)   - ellipsoid centre
    """
    
    # Write code here
    P_opt = None
    c_opt = None

    return P_opt, c_opt


# %% [markdown]
# The example code from above with the solver function applied in the middle.

# %%
modes = ['clusters', 'annulus', 'simplex', 'ellipse', 'cross']
n = 40
s = 2026

# Setting up plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Generating data
for i, mode in enumerate(modes):
    points = generate_points(mode=mode, n_points=n, seed=s)
    P, c = solve_mve_cvxpy(points)
    plot_mve(points, P, c, title=f'Mode: {mode}', ax=axes[i])

# Share common plot legend
axes[-1].axis('off')
handles, labels = axes[0].get_legend_handles_labels()
for ax in axes[:-1]:
    ax.get_legend().remove()
axes[-1].legend(handles, labels, fontsize=11, loc='center',
                title='Legend', title_fontsize=12,
                frameon=False)

# Finalise plot
plt.suptitle('Minimum Volume Enclosing Ellipsoid solved using CVXPY (SDP)', fontsize=13)
plt.tight_layout()
plt.show()
