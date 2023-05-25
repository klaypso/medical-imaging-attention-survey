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

# Got through the Excel
for d in datasets:
    print(d)
    df = xl.parse(d)
    plt.figure()
    for i, row in df.iterrows():
        legend = row['Model']
        row = row.drop("Model")
        accs = row.values
        plt.plot(x, accs, label=legend, color=colors[i], linestyle=style[i], marker=markers[i], markersize=4)
        plt.ylim(bottom=0