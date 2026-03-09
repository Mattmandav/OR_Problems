import random
import math
import gurobipy as gp
import itertools
import matplotlib.pyplot as plt

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
random.seed(2025)
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