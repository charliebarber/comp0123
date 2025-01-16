import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import os

input_files = []
for constellation in [
    "kuiper_630_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls",
    "starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls",
    "telesat_1015_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"
]:
    input_file = os.path.join("data", constellation, "2000ms_for_200s", "network_paths", "path_metrics.csv")
    input_files.append(input_file)

df0 = pd.read_csv(input_files[0])
df0['constellation'] = 'Kuiper'
df1 = pd.read_csv(input_files[1])
df1['constellation'] = 'Starlink'
df2 = pd.read_csv(input_files[2])
df2['constellation'] = 'Telesat'
combined_df = pd.concat([df0, df1, df2])

# Convert timestamps to seconds
combined_df['seconds'] = combined_df['timestamp'] / 1_000_000_000
combined_df['scaled_avg_shortest_path'] = combined_df['average_path_length'] / 1_000

# Create the plot
plt.figure(figsize=(12, 7))

sns.set_theme()
sns.set_style('white')
sns.despine()

# Create seaborn line plot
sns.lineplot(data=combined_df, 
            x='seconds', 
            y='scaled_avg_shortest_path',
            hue='constellation',  # Color by configuration
            style='constellation',  # Different markers for each configuration
)
            # markers=['o', 's', '^'],
            # markersize=8)

output_dir = os.path.join("figs")
os.makedirs(output_dir, exist_ok=True)

# Customize the plot
plt.xlabel('Time (seconds)')
plt.ylabel('Average Shortest Path Length (km)')
# plt.title('Temporal Degree Distribution')

# Adjust layout
plt.tight_layout()
plt.savefig(output_dir + "/avg_shortest_path.png")
