import matplotlib
matplotlib.use('Agg')  # Ensures no GUI is required for plotting
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set Seaborn theme
sns.set_theme()

def plot_graph_metrics(input_file, output_dir):
    # Read the CSV file
    df = pd.read_csv(input_file)
    # Convert timestamp to seconds
    df['seconds'] = df['timestamp'] / 1_000_000_000
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot average path length vs. k_value
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='k_value', y='average_path_length', hue='seconds', palette='coolwarm')
    plt.title('Average Path Length vs k-value')
    plt.xlabel('k-value (Nodes Removed)')
    plt.ylabel('Average Path Length')
    plt.legend(title='Time (seconds)', loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_path_length_vs_k.png"))
    plt.close()
    
    # 2. Plot connectivity percentage vs. k_value
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='k_value', y='connectivity_percentage', hue='seconds', palette='coolwarm')
    plt.title('Connectivity Percentage vs k-value')
    plt.xlabel('k-value (Nodes Removed)')
    plt.ylabel('Connectivity Percentage (%)')
    plt.legend(title='Time (seconds)', loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "connectivity_percentage_vs_k.png"))
    plt.close()
    
    # 3. Plot min and max path lengths vs. k_value
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='k_value', y='min_path_length', label='Min Path Length')
    sns.lineplot(data=df, x='k_value', y='max_path_length', label='Max Path Length')
    plt.title('Path Lengths vs k-value')
    plt.xlabel('k-value (Nodes Removed)')
    plt.ylabel('Path Lengths')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "path_lengths_vs_k.png"))
    plt.close()
    
    # 4. Plot average hops vs. k_value
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='k_value', y='average_hops', hue='seconds', palette='coolwarm')
    plt.title('Average Hops vs k-value')
    plt.xlabel('k-value (Nodes Removed)')
    plt.ylabel('Average Hops')
    plt.legend(title='Time (seconds)', loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_hops_vs_k.png"))
    plt.close()
    
    # 5. Boxplot for connectivity percentage grouped by seconds
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='seconds', y='connectivity_percentage', palette='Set2')
    plt.title('Connectivity Percentage Distribution Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Connectivity Percentage (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "connectivity_percentage_boxplot.png"))
    plt.close()

# Process datasets for multiple constellations
for constellation in [
    "kuiper_630",
    "starlink_550",
    "telesat_1015"
]:
    input_file = os.path.join("data", constellation, "2000ms_for_200s", "network_paths", "metrics.csv")
    output_dir = os.path.join("figs", constellation, "2000ms_for_200s")
    os.makedirs(output_dir, exist_ok=True)
    plot_graph_metrics(input_file, output_dir)
