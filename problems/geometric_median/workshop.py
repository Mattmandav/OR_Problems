## Required packages
import random
import gurobipy as gp
import matplotlib.pyplot as plt

# The value for m
number_of_houses = 10

# Generating points on a unit square
random.seed(2026)
a_values = []
for i in range(number_of_houses):
    a_values.append((random.random(), random.random()))

# Preview of data
for i in range(5):
    try:
        print(f"House {i+1} located at {a_values[i]}")
    except:
        pass