import re
import csv

def text_to_csv(input_file_path, output_file_path):
    # Read the text file
    with open(input_file_path, "r") as file:
        text_data = file.read()

    # Initialize CSV rows
    rows = [["K Value", "Mean Average Path Length (km)", "Change from Baseline", "Z-score", "Statistically Significant"]]

    # Extract k-value blocks and parse the data
    pattern = re.compile(
        r"K Value: (\d+).*?"
        r"Mean Average Path Length \(km\): ([\d.]+).*?"
        r"Change from baseline: ([\d.]+).*?"
        r"Z-score: ([\d.-]+).*?"
        r"Statistically significant change: (True|False)",
        re.S
    )

    for match in pattern.finditer(text_data):
        k_value = match.group(1)
        mean_length = match.group(2)
        change = match.group(3)
        z_score = match.group(4)
        significant = match.group(5)
        rows.append([k_value, mean_length, change, z_score, significant])

    # Write to CSV file
    with open(output_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

for constellation in ["kuiper_630","starlink_550","telesat_1015"]:
    dir = f"figs/{constellation}/5000ms_for_200s/"
    input_file_path = dir + "z_score_analysis_path_length.txt"
    output_file_path = dir + "z_score_analysis_path_length.csv"  # Replace with the desired output path for the CSV file
    text_to_csv(input_file_path, output_file_path)
