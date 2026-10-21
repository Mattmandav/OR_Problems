# %% [markdown]
# # Chebyshev Centre of a Polyhedron
#
# Let $P$ be a polyhedron described as $P = \{x \in \mathbb{R}^n : a_i^Tx \leq b_i, i = 1,\dots,m\}$. The centre of the largest ball is called the Chebyshev centre of the polyhedron and this ball $B$ can be described as $B = \{x_c + u : ||u||_2 \leq r\}$.
#
# ## Problem description
# The model variables are the following:
# * $x_c \in \mathbb{R}^n$: Centre of the ball $B$.
# * $r \in \mathbb{R}$: Radius of the ball $B$.
#
# The objective is:
#
# $\max r$,
#
# subject to:
#
# $a_i^Tx_c + r||a_i||_2 \leq b_i$.
#
# ## Required packages

# %%
# %matplotlib ipympl
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


# %% [markdown]
# ## Data generation
#
# The following function generates a random polyhedron of dimension $n$.

# %%
def generate_polyhedron(
        n = 2,
        m = 5,
        seed=2026
    ):
    rng = np.random.default_rng(seed)

    # Random halfspaces: directions drawn from unit sphere
    A_random = rng.standard_normal((m, n))
    A_random /= np.linalg.norm(A_random, axis=1, keepdims=True)
    b_random = rng.uniform(1.0, 2.0, size=m)  # offsets ensure origin is interior

    # Bounding box: -R <= x_i <= R, written as ±e_i^T x <= R
    R = 3.0
    A_box = np.vstack([np.eye(n), -np.eye(n)])
    b_box = R * np.ones(2 * n)

    A = np.vstack([A_random, A_box])
    b = np.hstack([b_random, b_box])

    return A, b


# %% [markdown]
# The following code plots the data for either $n=2$ or $n=3$

# %%
def plot_polyhedron(
        A,
        b,
        x_c=None,
        r=0,
        seed=2026
    ):
    n = A.shape[1]
    assert n in (2, 3), "Only n=2 and n=3 are supported."

    # Find vertices by dense sampling + feasibility check
    # Get bounding box from the box constraints (rows where A = ±I)
    R = np.max(np.abs(b[-2*n:]))  # bounding box radius
    N_samples = 10000000
    rng = np.random.default_rng(seed)
    samples = rng.uniform(-R, R, size=(N_samples, n))
    feasible = np.all(samples @ A.T <= b, axis=1)
    points = samples[feasible]
    if len(points) < 4:
        raise ValueError("Too few feasible points found — try increasing N_samples.")
    hull = ConvexHull(points)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') if n == 3 else fig.add_subplot(111)

    if n == 2:
        vertices = points[hull.vertices]
        # Sort vertices by angle for a clean polygon
        centre = vertices.mean(axis=0)
        angles = np.arctan2(vertices[:, 1] - centre[1], vertices[:, 0] - centre[0])
        vertices = vertices[np.argsort(angles)]
        poly = plt.Polygon(vertices, alpha=0.4, facecolor='steelblue', edgecolor='navy', linewidth=1.5)
        ax.add_patch(poly)
        if x_c is not None:
            plt.plot(x_c[0], x_c[1],"ko")
            circ = plt.Circle(x_c, r, color='r', alpha=0.4)
            ax.add_patch(circ)
        ax.set_xlim(-R * 1.1, R * 1.1)
        ax.set_ylim(-R * 1.1, R * 1.1)
        ax.set_aspect('equal')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title('Polyhedron ($n=2$)')

    elif n == 3:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        faces = []
        for simplex in hull.simplices:
            faces.append(points[simplex])
        poly3d = Poly3DCollection(faces, alpha=0.3, facecolor='steelblue', edgecolor='navy', linewidth=0.5)
        if x_c is not None:
            # Make sphere data
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = (r * np.outer(np.cos(u), np.sin(v))) + x_c[0]
            y = (r * np.outer(np.sin(u), np.sin(v))) + x_c[1]
            z = (r * np.outer(np.ones(np.size(u)), np.cos(v))) + x_c[2]
            ax.plot_surface(x, y, z)
        ax.add_collection3d(poly3d)
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_zlim(-R, R)
        ax.set_aspect('equal')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$x_3$')
        ax.set_title('Polyhedron ($n=3$)')

    plt.tight_layout()


# %% [markdown]
# The following code previews the data

# %%
# Generate data
dimension = 3
halfspaces = 5
[A,b] = generate_polyhedron(dimension, halfspaces, seed=2026)
plot_polyhedron(A, b, x_c=None, r=0)

# %% [markdown]
# ## Centre finding

# %%
