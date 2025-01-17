import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define function to read and combine data
def load_and_combine_data():
    combined_data = []
    constellations = ["kuiper_630", "starlink_550", "telesat_1015"]
    
    for constellation in constellations:
        dir = f"figs/{constellation}/5000ms_for_200s/"
        # input_file_path = dir + "z_score_analysis_path_length.csv"
        input_file_path = dir + "z_score_analysis_connectivity.csv"
        df = pd.read_csv(input_file_path)
        
        # Add constellation label
        df["Constellation"] = constellation.capitalize()
        combined_data.append(df)
    
    return pd.concat(combined_data, ignore_index=True)

# Load data
df_combined = load_and_combine_data()

# Plot Z-scores against K Value
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df_combined,
    x="K Value",
    y="Z-score",
    hue="Constellation",
    marker="o"
)

# Plot the Z-score threshold as a horizontal dotted line at y=-2
# plt.axhline(y=-2, color='red', linestyle='--', label="Z-Score Threshold (-2)")
plt.axhline(y=2, color='red', linestyle='--', label="Z-Score Threshold (2)")

# Add labels and title
plt.title("Z-Scores vs K Value with Threshold", fontsize=14)
plt.xlabel("K Value", fontsize=12)
plt.ylabel("Z-Score", fontsize=12)

# Add legend
plt.legend(title="Constellation", fontsize=10, title_fontsize=12)

# Save the plot
output_file = "figs/connectivity_z_score_analysis_zscore_plot.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
