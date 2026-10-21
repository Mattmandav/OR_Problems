# %% [markdown]
# # Floor planning
#
# ## Problem description
# A factory wishes to place $m$ machines into a rectangular warehouse (not yet built). They wish to minimise the size of this warehouse to minimise costs. $w_i$ represents the width of machine $i$ and $h_i$ represents the height of machine $i$.
#
# ## Problem formulation
#
# The model variables are the following:
# * $a_i \in \mathbb{R}^2$: Location of machine $i$ where $i \in \{1, \dots, m\}$.
# * $x \in \mathbb{R}^2$: Decision variable representing the size of the warehouse.
#
# The objective is:
#
# $\min f(z) = 2(x_1 + x_2)$,
#
# subject to:
#
# $a_{i,0} + w_i \leq x_1, \quad \forall i \in \{1, \dots, m\}.$
#
# $a_{i,1} + h_i \leq x_2, \quad \forall i \in \{1, \dots, m\}.$
#
# and for every pair $(i,j)$ with $i \neq j$, at least one of the following must hold:
#
# $a_{i,0} + w_i \leq a_{j,0}, \quad$ “$i$ to the left of $j$”
#
# $a_{j,0} + w_j \leq a_{i,0}, \quad$ “$j$ to the left of $i$”
#
# $a_{i,1} + h_i \leq a_{j,1}, \quad$ “$i$ below $j$”
#
# $a_{j,1} + h_j \leq a_{i,1}, \quad$ “$j$ below $i$”
#
# ## Required packages

# %%
import random
import math
import gurobipy as gp
import itertools
import matplotlib.pyplot as plt

# %% [markdown]
# ## Data generation
#
# The following code generates random "machines" (rectangular shapes). The value for `number_of_machines` can be changed. The value of `maximum_area` controls the area that machines can take, also bounding the lengths of the machines. The largest machine is a square with maximum area.

# %%
# The number of machines
number_of_machines = 5
maximum_area = 100

# Machine class
class Machine():
    def __init__(self, area, width, length):
        self.area = area
        self.width = width
        self.length = length

# Generating machines
random.seed(2026)
machines = {}
lengths = [i for i in range(1, int(math.sqrt(maximum_area))+1, 1)]
for i in range(number_of_machines):
    machine_width = random.choice(lengths)
    machine_length = random.choice(lengths)
    machine_area = machine_width*machine_length
    machines[len(machines)+1] = Machine(machine_area, machine_width, machine_length)

# Sample of data
for i in range(1,min(6,number_of_machines)):
    print(f"Machine {i}: Area={machines[i].area}, Width={machines[i].width}, Length={machines[i].length}")

# %% [markdown]
# ## Problem without rotation
#
# Find the smallest warehouse that can fit all of the machines, without rotation.

# %%

# %% [markdown]
# ## Problem without rotation with spacing
#
# Find the smallest warehouse that can fit all of the machines, without rotation. Ensure that there is at least $\rho$ metres between machines.

# %%

# %% [markdown]
# ## Problem with rotation
#
# Find the smallest warehouse that can fit all of the machines, with rotation.
