# %% [markdown]
# # Maximum cut
#
# ## Problem description
#
# Consider an undirected graph $G = (V, E)$ (assume $|V| = n$) with weights on the edges in $E$. The goal of the Max-Cut problem is to find a cut (partition nodes into $S$ and $V\setminus S$) so that the sum of the weights of the cut edges is maximised.
#
# ## Problem formulation 1
#
# The model variables are the following:
# * $v_i \in \{0,1\}$: Binary variable indicating if vertex $i$ is in $S$.
#
# The objective is:
#
# $\max f(v) = \sum_{(i,j)\in E} w_{ij} |v_i - v_j|$,
#
# where $w_{ij}$ is the weight of the edge connecting node $i$ with node $j$.
#
# ## Problem formulation 2
#
# The model variables are the following:
# * $v_i \in \{-1,1\}$: Integer variable indicating if vertex $i$ is in $S$ ($v_i = 1$) or not ($v_i = -1$).
#
# The objective is:
#
# $\max f(v) = \frac{1}{4}\sum_{i = 1}^n \sum_{j = 1}^n w_{ij} (1 - v_i\cdot v_j)$,
#
# where $w_{ij}$ is the weight of the edge connecting node $i$ with node $j$.
#
# We can relax the domain of the variables to be continuous and impose the following constraints:
#
# $v_i^2 = 1, \quad \forall i \in \{1, \dots, n\}.$
#
# ## Required packages

# %%
import random
import gurobipy as gp
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np
import scipy as sp
import cvxopt # needed as a solver backend

# %% [markdown]
# ## Data generation
#
# The following code generates random connected graph. The value for `number_of_nodes`, `edge_density` and `max_edge_weight` can be changed.

# %%
# The values for the graph
number_of_nodes = 10
edge_density = 0.5      # Should be in [0,1]
max_edge_weight = 10
random.seed(2026)

# Correcting edge_density if not in [0,1]
edge_density = max(min(edge_density, 1), 1/number_of_nodes)
# Creating edge weight options
edge_weights = [i for i in range(1,max(1,max_edge_weight))]

# Generating nodes of graph
G = nx.Graph()
for i in range(number_of_nodes):
    G.add_node(i)

# Creating set of all connections
edge_set = set(combinations(range(number_of_nodes), 2))

# Connecting all nodes (unconnected components are boring)
for i in range(number_of_nodes-1):
    G.add_edge(i, i+1, weight=random.choice(edge_weights))
    edge_set.remove((i,i+1))

# Randomly choose edges to add until density reached
while True:
    edge = random.choice(list(edge_set))
    G.add_edge(edge[0],edge[1], weight=random.choice(edge_weights))
    edge_set.remove(edge)
    if nx.density(G) > edge_density:
        break

# %% [markdown]
# The following code provides a preview of the graph.

# %%
pos = nx.spring_layout(G)
nx.draw(G, pos)

# %% [markdown]
# # Semidefinite relaxation
#
# The SDP relation of the maximum cut problem looks as follows:
#
# $\max f(v) = \frac{1}{4}\sum_{i = 1}^n \sum_{j = 1}^n w_{ij} (1 - \overline{\underline{X}}_{ij})$,
#
# where $w_{ij}$ is the weight of the edge connecting node $i$ with node $j$.
#
# Subject to:
#
# $\overline{\underline{X}} \in S_+^n,$
#
# $\overline{\underline{X}}_{ii} = 1, \quad \forall i \in \{1, \dots, n\}.$
