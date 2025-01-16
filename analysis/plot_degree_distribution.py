import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import os

def plot_degree_distribution(input_file, output_dir):
    # Read the CSV data
    df = pd.read_csv(input_file)

    # Convert timestamp to seconds
    df['seconds'] = df['timestamp'] / 1_000_000_000

    # Create a pivot table for the heatmap
    pivot_data = df.pivot(index='seconds', 
                      columns='degree', 
                      values='frequency_normalized')

    # Sort index in descending order to flip the y-axis
    pivot_data = pivot_data.sort_index(ascending=False)

    # Create the plot
    plt.figure(figsize=(15, 8))
    sns.set_theme()
    sns.color_palette()

    # Create heatmap with symlog-normalized colors
    sns.heatmap(pivot_data, 
              cmap='magma',
              norm=SymLogNorm(linthresh=0.01, 
                            vmin=pivot_data.values.min(),
                            vmax=pivot_data.values.max()),
              cbar_kws={'label': 'Normalized Frequency (symlog scale)'},
              yticklabels=sorted(range(10), reverse=True))  # Reverse the y-tick labels

    # Customize y-axis ticks to show every second
    seconds = pivot_data.index.unique()
    # Create ticks every second (or at desired intervals)
    tick_positions = np.arange(0, len(seconds), 20)  # Adjust 20 to change tick frequency
    tick_labels = [f"{seconds[i]:.1f}s" for i in tick_positions if i < len(seconds)]
    plt.yticks(tick_positions, tick_labels, rotation=0)

    # Customize the plot
    plt.xlabel('Node Degree')
    plt.ylabel('Time (seconds)')
    # plt.title('Temporal Degree Distribution')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(output_dir + "/degree_distribution.png")

for constellation in [
    "kuiper_630_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls",
    "starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls",
    "telesat_1015_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"
]:
    input_file = os.path.join("data", constellation, "1000ms_for_200s", "network_properties", "degree_distributions.csv")
    output_dir = os.path.join("figs", constellation, "1000ms_for_200s")
    os.makedirs(output_dir, exist_ok=True)

    plot_degree_distribution(input_file, output_dir)