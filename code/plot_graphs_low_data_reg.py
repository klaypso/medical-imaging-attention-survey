# Imports
import pandas as pd
import matplotlib.pyplot as plt



# Read Excel
xl = pd.ExcelFile("results/experiments.xlsx")

# Get datasets' names
datasets = xl.sheet_names

# Parameters for the plot
x = [1, 2, 3, 4]
x_labels = ["1%", "10%", "50%", "100%"]
colors = ['C0', 'C1', 'C0', 'C1', 'C0', 'C1', 'C2']
style = ['solid', 'solid', 'dashed', 'dashed', 'dotted', 'dotted', 'solid']
markers = ['s', 'D', 's', 'D', 's', 'D', 'o']

# Got through the