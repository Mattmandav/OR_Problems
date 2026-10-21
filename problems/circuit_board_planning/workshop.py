# %% [markdown]
# # Circuit board planning
#
# ## Problem description
# A factory wishes to place $m$ chips onto a rectangular board. They wish to minimise the size of this board to minimise costs. $w_i$ represents the width of chip $i$ and $h_i$ represents the height of chip $i$.
#
# ## Problem formulation
#
# Problem formulation similar to floor planning problem.
#
# ## Required packages

# %%
import random
import math
import gurobipy as gp
import itertools
import matplotlib.pyplot as plt
import networkx as nx

# %% [markdown]
# ## Data generation
#
# The following code generates random "machines" (rectangular shapes). The value for `number_of_machines` can be changed. The value of `maximum_area` controls the area that machines can take, also bounding the lengths of the machines. The largest machine is a square with maximum area.

# %%
# The number of machines
number_of_chips = 4
maximum_area = 100
trace_spacing = 1
chip_spacing = 2

# Chip class
class Chip():
    def __init__(self, area, width, length):
        self.area = area
        self.width = width
        self.length = length

# Generating machines
random.seed(2026)
chips = {}
lengths = [i for i in range(1, int(math.sqrt(maximum_area))+1, 1)]
for i in range(number_of_chips):
    chip_width = random.choice(lengths)
    chip_length = random.choice(lengths)
    chip_area = chip_width*chip_length
    chips[len(chips)+1] = Chip(chip_area, chip_width, chip_length)

# Sample of data
for i in range(1,min(6,number_of_chips+1)):
    print(f"Chip {i}: Area={chips[i].area}, Width={chips[i].width}, Length={chips[i].length}")

# Required connections
attempts = 0
max_attempts = 20
while True:
    G = nx.Graph()
    for chip in chips:
        G.add_node(chip)
    connections_required = []
    all_possible_connections = list(itertools.combinations([chip for chip in chips], r=2))
    random.shuffle(all_possible_connections)
    for pair in all_possible_connections:
        G.add_edge(pair[0], pair[1])
        connections_required.append(pair)
        # Check everything connected
        if nx.is_connected(G):
            break
    # Check if planar (feasible)
    if nx.is_planar(G):
        break
    attempts += 1
    if max_attempts == attempts:
        raise Exception(f"Could not construct planar graph in {max_attempts} tries. Use different seed.")

# Show connections
nx.draw(G, pos=nx.planar_layout(G), with_labels=True)

# Using the planar graph as a start solution
smallest_negative = 0
plane_scaling = 100
planar_layout = nx.planar_layout(G)
for nd in planar_layout:
    for coord in planar_layout[nd]:
        if coord < smallest_negative:
            smallest_negative = coord
new_planar_layout = {}
for nd in planar_layout:
    new_planar_layout[nd] = (
        plane_scaling*(planar_layout[nd][0]+abs(smallest_negative)),
        plane_scaling*(planar_layout[nd][1]+abs(smallest_negative)),
        )

# %% [markdown]
# ## Problem without rotation
#
# Find the board that can fit all of the chips, without rotation, and adding all of the required traces

# %%

# %% [markdown]
# ## Plot the board layout

# %%
