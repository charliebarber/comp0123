#!/usr/bin/env python3
"""
Network Resilience Analysis and Plotting Script with Z-Score Analysis
"""

import matplotlib
matplotlib.use('Agg')  # Ensures no GUI is required for plotting
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional

# Set Seaborn theme
sns.set_theme()

def calculate_z_scores(df: pd.DataFrame, 
                      metric: str) -> Tuple[Dict, Dict]:
    """
    Calculate z-scores comparing each k value's effect against baseline (k=0).
    Uses a small epsilon for zero standard deviation cases to maintain z-score scale.
    """
    z_scores_dict = {}
    significant_dict = {}
    
    # Get baseline (k=0) statistics
    baseline_data = df[df['k_value'] == 0][metric]
    baseline_mean = baseline_data.mean()
    baseline_std = baseline_data.std()
    
    print(f"\nBaseline Statistics for {metric}:")
    print(f"  Mean: {baseline_mean:.2f}")
    print(f"  Std:  {baseline_std:.2f}")
    
    # If std is zero, use a small epsilon based on the mean
    if baseline_std == 0 or np.isnan(baseline_std):
        baseline_std = baseline_mean * 0.001  # Use 0.1% of mean as std
        print(f"Note: Using small standard deviation ({baseline_std:.2f}) for z-score calculation")
    
    # Calculate z-scores for each k value relative to baseline
    for k in sorted(df['k_value'].unique()):
        if k == 0:  # Skip baseline
            continue
            
        k_data = df[df['k_value'] == k][metric]
        k_mean = k_data.mean()
        
        # Calculate z-score using baseline statistics
        z_score = (baseline_mean - k_mean) / baseline_std
        z_scores_dict[k] = z_score
        significant_dict[k] = abs(z_score) > 2
        
        print(f"k={k:3d} - Mean: {k_mean:8.2f}, Z-score: {z_score:8.2f}, Significant: {significant_dict[k]}")
    
    return z_scores_dict, significant_dict

def plot_metric_over_time(df: pd.DataFrame, 
                         x: str, 
                         y: str, 
                         hue: str, 
                         output_path: str, 
                         title: str, 
                         xlabel: str, 
                         ylabel: str) -> None:
    """
    Plot a metric over time without smoothing.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=x, y=y, hue=hue, palette='tab20', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_z_score_analysis(df: pd.DataFrame,
                         z_scores: Dict,
                         significant_dict: Dict,
                         metric: str,
                         output_path: str,
                         title: str) -> None:
    """
    Create a visualization of the z-score analysis.
    """
    if z_scores is None or significant_dict is None:
        print(f"Skipping z-score plot for {metric} - invalid z-scores")
        return
        
    plt.figure(figsize=(12, 6))
    
    # Plot z-scores for each k value
    k_values = sorted(z_scores.keys())
    z_score_values = [z_scores[k] for k in k_values]
    
    # Create the bar plot
    bars = plt.bar(k_values, z_score_values, alpha=0.6)
    
    # Add significance threshold lines
    plt.axhline(y=2, color='r', linestyle='--', alpha=0.3, label='Significance Threshold')
    plt.axhline(y=-2, color='r', linestyle='--', alpha=0.3)
    
    # Highlight significant bars
    for k, bar in zip(k_values, bars):
        if significant_dict[k]:
            bar.set_color('red')
            bar.set_alpha(0.6)
    
    plt.title(f"{title}\nZ-Score Analysis")
    plt.xlabel('k Value')
    plt.ylabel('Z-Score (Relative to Baseline)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def output_z_score_analysis(z_scores: Dict, 
                          significant_dict: Dict, 
                          df: pd.DataFrame,
                          metric: str,
                          metric_name: str,
                          output_dir: str) -> None:
    """
    Output z-score analysis results to a text file.
    
    Args:
        z_scores: Dictionary of z-scores for each k value
        significant_dict: Dictionary indicating significance for each k value
        df: Input dataframe
        metric: Column name of the metric in the dataframe
        metric_name: Human-readable name of the metric for output
        output_dir: Directory to save the analysis file
    """
    """
    Output z-score analysis results to a text file.
    """
    if z_scores is None or significant_dict is None:
        return
        
    # Create more readable filename
    metric_filename = "path_length" if metric == "scaled_avg_shortest_path" else "connectivity"
    analysis_path = os.path.join(output_dir, f"z_score_analysis_{metric_filename}.txt")
    print(f"Writing z-score analysis to: {analysis_path}")
    
    # Get baseline statistics
    baseline_data = df[df['k_value'] == 0][metric]
    baseline_mean = baseline_data.mean()
    baseline_std = baseline_data.std()
    
    with open(analysis_path, 'w') as f:
        f.write(f"Z-Score Analysis Results for {metric_name}\n")
        f.write("=" * (32 + len(metric_name)) + "\n\n")
        
        f.write(f"Baseline Statistics (k=0):\n")
        f.write(f"  Mean {metric_name}: {baseline_mean:.2f}\n")
        f.write(f"  Standard Deviation: {baseline_std:.2f}\n\n")
        
        f.write("Analysis by k value:\n")
        f.write("-------------------\n")
        
        for k in sorted(z_scores.keys()):
            k_data = df[df['k_value'] == k][metric]
            k_mean = k_data.mean()
            
            f.write(f"\nK Value: {k}\n")
            f.write("---------\n")
            f.write(f"Mean {metric_name}: {k_mean:.2f}\n")
            f.write(f"Change from baseline: {k_mean - baseline_mean:.2f}\n")
            f.write(f"Z-score: {z_scores[k]:.2f}\n")
            f.write(f"Statistically significant change: {significant_dict[k]}\n")
            
        f.write("\nInterpretation:\n")
        f.write("--------------\n")
        significant_ks = [k for k, is_sig in significant_dict.items() if is_sig]
        if significant_ks:
            f.write(f"First significant change occurs at k={min(significant_ks)}\n")
            f.write("Significant changes in k values: " + 
                   ", ".join(str(k) for k in sorted(significant_ks)) + "\n")
        else:
            f.write("No statistically significant changes detected\n")

def plot_metrics(input_file: str, output_dir: str) -> None:
    """
    Process and plot all metrics for a given constellation.
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Convert timestamp to seconds and scale path length
    df['seconds'] = df['timestamp'] / 1_000_000_000
    df['scaled_avg_shortest_path'] = df['average_path_length'] / 1_000
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze connectivity percentage
    print("\nAnalyzing Connectivity Percentage...")
    z_scores_conn, significant_dict_conn = calculate_z_scores(df, 'connectivity_percentage')
    
    plot_z_score_analysis(
        df,
        z_scores_conn,
        significant_dict_conn,
        'connectivity_percentage',
        os.path.join(output_dir, "connectivity_z_scores.png"),
        'Connectivity Percentage'
    )
    
    output_z_score_analysis(
        z_scores_conn,
        significant_dict_conn,
        df,
        'connectivity_percentage',
        'Connectivity Percentage',
        output_dir
    )
    
    # Analyze average path length
    print("\nAnalyzing Average Path Length...")
    z_scores_path, significant_dict_path = calculate_z_scores(df, 'scaled_avg_shortest_path')
    
    if z_scores_path is not None and significant_dict_path is not None:
        plot_z_score_analysis(
            df,
            z_scores_path,
            significant_dict_path,
            'scaled_avg_shortest_path',
            os.path.join(output_dir, "path_length_z_scores.png"),
            'Average Path Length'
        )
        
        output_z_score_analysis(
            z_scores_path,
            significant_dict_path,
            df,
            'scaled_avg_shortest_path',
            'Average Path Length (km)',
            output_dir
        )
    else:
        print("Warning: Could not calculate z-scores for average path length")
    
    # Plot regular time series
    print("\nGenerating time series plots...")
    plot_metric_over_time(
        df=df,
        x='seconds',
        y='connectivity_percentage',
        hue='k_value',
        output_path=os.path.join(output_dir, "connectivity_percentage_over_time.png"),
        title='Connectivity Percentage Over Time',
        xlabel='Time (seconds)',
        ylabel='Connectivity Percentage (%)'
    )
    
    plot_metric_over_time(
        df=df,
        x='seconds',
        y='scaled_avg_shortest_path',
        hue='k_value',
        output_path=os.path.join(output_dir, "average_path_length_over_time.png"),
        title='Average Path Length Over Time',
        xlabel='Time (seconds)',
        ylabel='Average Path Length (km)'
    )

def main():
    """Main execution function"""
    # Process datasets for multiple constellations
    constellations = [
        "kuiper_630",
        "starlink_550",
        "telesat_1015"
    ]
    
    for constellation in constellations:
        print(f"\nProcessing {constellation}...")
        input_file = os.path.join(
            "data", 
            constellation, 
            "5000ms_for_200s", 
            "network_paths", 
            "resilience_metrics.csv"
        )
        
        if not os.path.exists(input_file):
            print(f"Warning: Input file not found: {input_file}")
            continue
            
        output_dir = os.path.join(
            "figs", 
            constellation, 
            "5000ms_for_200s"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            plot_metrics(input_file, output_dir)
            print(f"Successfully processed {constellation}")
        except Exception as e:
            print(f"Error processing {constellation}: {str(e)}")

if __name__ == "__main__":
    main()