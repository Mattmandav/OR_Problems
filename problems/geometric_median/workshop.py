# %% [markdown]
# # Geometric Median - Facility Location
#
# ## Problem description
# A supermarket chain wishes to build a store in a small town. There are $m \in \mathbb{N}$ houses that need to be served and these are distributed randomly on a 2D plane. The supermarket owners want to minimise the sum of the distances between the new store and the houses.
#
# ## Problem formulation
#
# The model variables are the following:
# * $a_i \in \mathbb{R}^2$: Location of house $i$ where $i \in \{1, \dots, m\}$.
# * $x \in \mathbb{R}^2$: Decision variable representing the location of the new store.
# * $z_i \in \mathbb{R}$: Decision variable representing the distance of the new store from house $i$ where $i \in \{1, \dots, m\}$.
#
# The objective is:
#
# $\min f(z) = \sum_{i=1}^m z_i$,
#
# subject to:
#
# $||a_i - x||_2 \leq z_i, \quad \forall i \in \{1, \dots, m\}.$
#
# ## Required packages

# %%
import random
import gurobipy as gp
import matplotlib.pyplot as plt

# %% [markdown]
# ## Data generation
#
# The following code generates random house locations. The value for `number_of_houses` can be changed.

# %%
# The value for m
number_of_houses = 10

# Generating points on a unit square
random.seed(2026)
a_values = []
for i in range(number_of_houses):
    a_values.append((random.random(), random.random()))

# %% [markdown]
# The following code provides a preview of the data.

# %%
for i in range(5):
    try:
        print(f"House {i+1} located at {a_values[i]}")
    except:
        pass
