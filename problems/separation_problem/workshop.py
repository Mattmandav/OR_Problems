# %% [markdown]
# # Classification - Separation problem
#
# ## Problem description
# There are two sets of points $\{x_1,\dots,x_N\}$ and $\{y_1,\dots,y_M\}$. We would like to find a function, $f$, such that 
#
# $f(x_i) > 0, \quad \forall i = 1,\dots,N$
#
# $f(y_i) < 0, \quad \forall i = 1,\dots,M$
#
# ## Required packages

# %%
import random
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt


# %% [markdown]
# ## Data generation
#
# The following function generates two random sets of data. The number of points and the overlap of the two sets can be controlled.

# %%
# Generating points on unit square, separating around y = x
def data_generation(
        number_of_points,
        maximum_overlap,
        minimum_spacing = 0,
        seed = 2026,
    ):
    random.seed(seed)
    X = []
    Y = []
    minimum_spacing = min(minimum_spacing,1)
    for i in range(number_of_points):
        # Generate point
        while True:
            point = (random.random(), random.random())
            closest_point = (point[0]+point[1])/2
            distance = np.linalg.norm(tuple([point[i] - closest_point for i in range(2)]))
            # Check spacing if fine
            if distance > minimum_spacing/2: 
                # Probability of not going in expected set
                prob = 0
                if distance <= maximum_overlap:
                    prob = (1 - distance/maximum_overlap)*0.5
                break
        # Deciding if it switches set or not
        switch = False
        roll = random.random()
        if roll < prob:
            switch = True
        # Place point in set
        if point[0] > point[1]:
            if switch:
                Y.append(point)
            else:
                X.append(point)
        elif point[0] < point[1]:
            if switch:
                X.append(point)
            else:
                Y.append(point)
        else:
            if random.random() < 0.5:
                Y.append(point)
            else:
                X.append(point)
    # Return points
    return X,Y


# %% [markdown]
# The following code previews the data

# %%
# Generate data
number_of_points = 20
maximum_overlap = 0.0
minimum_spacing = 0.0
[X,Y] = data_generation(number_of_points,maximum_overlap,minimum_spacing)

# Plot the data and the line y = x
plt.plot([x[0] for x in X], [x[1] for x in X], 'ro')
plt.plot([y[0] for y in Y], [y[1] for y in Y], 'bo')
plt.plot([i for i in np.linspace(0,1,100)], [i for i in np.linspace(0,1,100)], 'k-')

# Show the plot
plt.show()

# %% [markdown]
# ## Linear discrimination main

# %%

# %% [markdown]
# ## Linear discrimination alternative

# %%

# %% [markdown]
# ## Robust linear discrimination

# %%

# %% [markdown]
# ## Support vector classifier

# %%
