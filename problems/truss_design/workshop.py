# %% [markdown]
# # Truss Topology Design: Minimum Compliance to SDP
#
# ## Problem Setup
#
# A truss consists of:
# - A set of **nodes** $V \subset \mathbb{R}^2$, some fixed and some free
# - A set of **bars** $B$, each connecting two nodes
# - A **load vector** $f \in \mathbb{R}^n$, where $n$ is the number of free degrees of freedom
#
# The **design variables** are the bar volumes $t_i \geq 0$ for $i = 1, \dots, m$,
# subject to a total volume constraint $\sum_{i=1}^m t_i = V$.

# %%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cvxpy as cp


# %% [markdown]
# The following code builds the 6-node toy truss from Ben-Tal & Nemirovski (1997), Figure 2.1.
# The fixed nodes can be specified by a set and a load vector is given in the form `{(<node>,<"X" or "Y">,<load>)}`.

# %%
def build_toy_truss(
        fixed_nodes = set(),
        load_vector = set(),   
    ):
    # Node positions (x, y)
    node_positions = {
        1: (0.0, 0.0),
        2: (1.0, 0.0),
        3: (2.0, 0.0),
        4: (0.0, 1.0),
        5: (1.0, 1.0),
        6: (2.0, 1.0),
    }

    free_nodes  = {1, 2, 3, 4, 5, 6} - fixed_nodes
    sorted_free_nodes = sorted(list(free_nodes))
    free_node_dof_index = {v:2*i for i,v in enumerate(sorted_free_nodes)}

    # All possible bars between nodes (ground structure = complete graph)
    all_bars = [(i, j) for i in range(1, 7) for j in range(i+1, 7)]
    load_to_f = dict()
    n_dof = 0
    for i in range(len(free_nodes)):
        for dim in ["X", "Y"]:
            load_to_f[(sorted_free_nodes[i],dim)] = n_dof
            n_dof += 1
    f = np.zeros(n_dof)
    
    # Add relevant loads
    for load in load_vector:
        f[load_to_f[(load[0],load[1])]] = load[2]

    # Normalise to unit Euclidean norm
    f = f / np.linalg.norm(f)

    # Return everything
    return load_vector, node_positions, fixed_nodes, free_nodes, all_bars, f, n_dof, free_node_dof_index


# %% [markdown]
# Function that can plot the ground structure with nodes, bars, and loads. If optimal bar volume is given then this is used to illustrate solution.

# %%
def plot_ground_structure(load_vector, node_positions, fixed_nodes, free_nodes, all_bars,
                          t_opt=None, tau_opt=None, threshold=1e-3, title_string = ""):
    G = nx.Graph()
    G.add_nodes_from(node_positions.keys())
    G.add_edges_from(all_bars)
    fig, ax = plt.subplots(figsize=(7, 5))

    # Draw bars
    if t_opt is not None:
        # Draw optimal bars with thickness proportional to volume
        max_vol = max(t_opt.values())
        for (p, q), vol in t_opt.items():
            x_vals = [node_positions[p][0], node_positions[q][0]]
            y_vals = [node_positions[p][1], node_positions[q][1]]
            if vol >= threshold:
                lw = 8 * vol / max_vol
                ax.plot(x_vals, y_vals, color='steelblue', 
                        lw=lw, solid_capstyle='round', zorder=2)
            else:
                # Draw inactive bars faintly as ground structure
                ax.plot(x_vals, y_vals, color='lightsteelblue', 
                        lw=0.5, alpha=0.4, zorder=1)
    else:
        # Draw full ground structure
        nx.draw_networkx_edges(
            G, pos=node_positions, ax=ax,
            edge_color='lightsteelblue', width=1.0, alpha=0.7
        )

    # Draw fixed nodes
    nx.draw_networkx_nodes(
        G, pos=node_positions, ax=ax,
        nodelist=list(fixed_nodes),
        node_color='black', node_size=200, node_shape='s', label='Fixed (#)'
    )

    # Draw free nodes
    nx.draw_networkx_nodes(
        G, pos=node_positions, ax=ax,
        nodelist=list(free_nodes),
        node_color='steelblue', node_size=200, node_shape='o', label='Free (*)'
    )

    # Node labels
    nx.draw_networkx_labels(
        G, pos=node_positions, ax=ax,
        font_color='white', font_size=8, font_weight='bold'
    )

    # Load arrows
    load_arrows = {}
    for load in load_vector:
        node, axis, mag = load
        if axis == "X":
            load_arrows[node] = (0.25 * np.sign(mag), 0)
        elif axis == "Y":
            load_arrows[node] = (0, 0.25 * np.sign(mag))

    for node, (dx, dy) in load_arrows.items():
        x, y = node_positions[node]
        ax.annotate(
            '', xy=(x + dx, y + dy), xytext=(x, y),
            arrowprops=dict(arrowstyle='->', color='crimson', lw=2)
        )

    # Ground supports
    for node in fixed_nodes:
        x, y = node_positions[node]
        ax.plot([x - 0.1, x + 0.1], [y - 0.08, y - 0.08], 'k-', lw=2)
        for dx in np.linspace(-0.1, 0.1, 5):
            ax.plot([x + dx, x + dx - 0.05], [y - 0.08, y - 0.16], 'k-', lw=1)

    # Legend
    fixed_patch = mpatches.Patch(color='black', label='Fixed node (#)')
    free_patch  = mpatches.Patch(color='steelblue', label='Free node (*)')
    load_patch  = mpatches.Patch(color='crimson', label='Applied load')
    ax.legend(handles=[fixed_patch, free_patch, load_patch], 
              loc='upper left', fontsize=9)

    # Title
    if tau_opt is not None:
        title = f'Optimal Truss - Compliance = {tau_opt:.4f}'
    else:
        if len(title_string) > 0:
            title = f'Ground Structure - {title_string}'
        else:
            title = 'Ground Structure'
    ax.set_title(title, fontsize=11)

    ax.set_xlim(-0.4, 2.6)
    ax.set_ylim(-0.4, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# %% [markdown]
# Example from Ben-Tal & Nemirovski (1997)

# %%
[load_vector, node_positions, fixed_nodes, free_nodes, all_bars, f, n_dof, free_node_dof_index] = build_toy_truss(
    fixed_nodes={1,4}, 
    load_vector = {(5, "Y", 0.5), (6, "X", 0.5), (3, "Y", -0.5), (2, "X", -0.5)}
)
plot_ground_structure(load_vector, node_positions, fixed_nodes, free_nodes, all_bars, title_string = "Ben-Tal & Nemirovski (1997), Fig. 2.1")


# %% [markdown]
# ## Bar Stiffness Matrix
#
# For bar $i$ connecting nodes $p$ and $q$ with positions $x_p, x_q \in \mathbb{R}^2$,
# the unit direction vector is:
#
# $e_i = (x_q - x_p)/\|x_q - x_p\|$
#
# The bar force vector $b_i \in \mathbb{R}^n$ is assembled into the global DOF space as:
#
# $b_i = \frac{1}{L_i} \begin{cases} +e_i & \text{at DOFs of node } p \text{ (if free)} \\ -e_i & \text{at DOFs of node } q \text{ (if free)} \\ 0 & \text{at DOFs of fixed nodes} \end{cases}$
#
# where $L_i = \|x_q - x_p\|$ is the bar length. Each bar stiffness matrix is:
#
# $A_i = b_i b_i^T \in \mathbb{R}^{n \times n}$
#
# which is **rank-1 positive semidefinite** by construction.

# %%
def bar_stiffness_matrix(node_p, node_q, pos_p, pos_q, free_node_dof_index, n_dof):
    diff = pos_q - pos_p
    L = np.linalg.norm(diff)
    e = diff / (L * L)

    b_i = np.zeros(n_dof)

    if node_p in free_node_dof_index:
        idx = free_node_dof_index[node_p]
        b_i[idx:idx+2] += e

    if node_q in free_node_dof_index:
        idx = free_node_dof_index[node_q]
        b_i[idx:idx+2] -= e

    A_i = np.outer(b_i, b_i)

    return A_i, b_i

def construct_bar_matrices(node_positions, all_bars, n_dof, free_node_dof_index):
    bar_matrices = {}
    for (p, q) in all_bars:
        pos_p = np.array(node_positions[p], dtype=float)
        pos_q = np.array(node_positions[q], dtype=float)
        A_i, b_i = bar_stiffness_matrix(
            p, q, pos_p, pos_q, free_node_dof_index, n_dof
        )
        bar_matrices[(p, q)] = (A_i, b_i)
    return bar_matrices


# %% [markdown]
# ## Global Stiffness Matrix
#
# The global stiffness matrix is a weighted sum of bar stiffness matrices:
#
# $A(t) = \sum_{i=1}^m t_i A_i$
#
# Since each $A_i \succeq 0$ and $t_i \geq 0$, we have $A(t) \succeq 0$ for all feasible $t$.
#
# ## Minimum Compliance Problem
#
# Given load $f$ and total volume $V$, the **minimum compliance problem** is:
#
# $\min_{t} f^T A(t)^{-1} f$
#
# subject to
#
# $\sum_{i=1}^m t_i = V$
#
# $t_i \geq 0, \quad i = 1, \dots, m$
#
# The objective $f^T A(t)^{-1} f$ is the **compliance**. It measures the work done by
# the load, or equivalently the inverse of the stiffness. A stiffer truss has lower compliance.
#
# This problem is **not** in a standard convex form because of the matrix inverse in the
# objective.
#
# ## SDP Formulation
#
# $\min_{t,\ \tau} \tau$
#
# subject to
#
# $\begin{pmatrix} \tau & f^T \\ f & A(t) \end{pmatrix} \succeq 0$
#
# $\sum_{i=1}^m t_i = V$
#
# $t_i \geq 0, \quad i = 1, \dots, m$
#
# where $A(t) = \sum_{i=1}^m t_i A_i$ is linear in $t$.

# %%
def solve_truss_sdp(bar_matrices, f, n_dof, all_bars, fixed_nodes):

    """
    Fill in code here
    """

    # Extract results
    t_opt   = dict()  # {bar: t[i].value for i, bar in enumerate(active_bars)}
    tau_opt = float() # value of tau

    return t_opt, tau_opt


# %% [markdown]
# ## Test problem from paper

# %%
# Input the problem from paper
[load_vector, node_positions, fixed_nodes, free_nodes, all_bars, f, n_dof, free_node_dof_index] = build_toy_truss(
    fixed_nodes={1,4}, 
    load_vector = {(5, "Y", 1), (6, "X", 1), (3, "Y", -1), (2, "X", -1)}
)
bar_matrices = construct_bar_matrices(node_positions, all_bars, n_dof, free_node_dof_index)

# Solve the problem
t_opt, tau_opt = solve_truss_sdp(bar_matrices, f, n_dof, all_bars, fixed_nodes)

# Extract the solution from the sdp and compare with paper
print(f"Optimal compliance: {tau_opt:.4f}")
print(f"Expected compliance (from paper): 16.000\n")
print("Bar volumes (active bars only):")
for (p, q), vol in sorted(t_opt.items(), key=lambda x: -x[1]):
    if vol > 1e-3:
        print(f"  Bar {p}-{q}: {vol:.4f}  ({100*vol:.2f}%)")
plot_ground_structure(load_vector, node_positions, fixed_nodes, free_nodes, all_bars, t_opt=t_opt, tau_opt=tau_opt)

# %%
# Other interesting loads
load_vector = {(5, "Y", 1), (6, "X", 1)}
load_vector = {(5, "Y", -1), (6, "X", 1)}
load_vector = {(5, "Y", -1), (6, "X", -1)}
load_vector = {(5, "Y", -1), (6, "Y", -1)}
