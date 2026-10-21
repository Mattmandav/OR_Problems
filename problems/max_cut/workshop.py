## Required packages
import random
import gurobipy as gp
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np
import scipy as sp
import cvxopt # needs installed for optimization

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

# The following code provides a preview of the graph.
pos = nx.spring_layout(G)
nx.draw(G, pos)