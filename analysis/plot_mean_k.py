import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def combine_csvs_and_plot(constellations, output_dir):
    # DataFrames for combined data
    path_length_df = pd.DataFrame()
    connectivity_df = pd.DataFrame()

    # Combine data from each constellation
    for constellation in constellations:
        dir = f"figs/{constellation}/5000ms_for_200s/"
        
        # Path length data
        path_length_file = dir + "z_score_analysis_path_length.csv"
        path_length_data = pd.read_csv(path_length_file)
        path_length_data["Constellation"] = constellation  # Add constellation identifier
        path_length_df = pd.concat([path_length_df, path_length_data], ignore_index=True)

        # Connectivity data
        connectivity_file = dir + "z_score_analysis_connectivity.csv"
        connectivity_data = pd.read_csv(connectivity_file)
        connectivity_data["Constellation"] = constellation  # Add constellation identifier
        connectivity_df = pd.concat([connectivity_df, connectivity_data], ignore_index=True)

    # Plot Mean Average Path Length
    plot_seaborn_line_chart(
        df=path_length_df,
        x_col="K Value",
        y_col="Mean Average Path Length (km)",
        hue_col="Constellation",
        title="Mean Average Path Length vs K Value",
        y_label="Mean Average Path Length (km)",
        x_label="K Value",
        output_path=f"{output_dir}combined_path_length_plot.png"
    )

    # Plot Mean Connectivity Percentage
    plot_seaborn_line_chart(
        df=connectivity_df,
        x_col="K Value",
        y_col="Mean Connectivity Percentage",
        hue_col="Constellation",
        title="Mean Connectivity Percentage vs K Value",
        y_label="Mean Connectivity Percentage",
        x_label="K Value",
        output_path=f"{output_dir}combined_connectivity_plot.png"
    )

def plot_seaborn_line_chart(df, x_col, y_col, hue_col, title, y_label, x_label, output_path):
    # Create a Seaborn lineplot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, marker="o")

    # Chart details
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(title=hue_col, fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path)
    plt.close()

# Example usage
constellations = ["kuiper_630", "starlink_550", "telesat_1015"]
output_directory = "figs/"
combine_csvs_and_plot(constellations, output_dir=output_directory)
