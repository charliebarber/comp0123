import re
import csv

def text_to_csv(input_file_path, output_file_path):
    with open(input_file_path, "r") as file:
        text_data = file.read()

    rows = [["K Value", "Mean Connectivity Percentage", "Change from Baseline", "Z-score", "Statistically Significant"]]

    pattern = re.compile(
        r"K Value: (\d+).*?"
        r"Mean Connectivity Percentage: ([\d.]+).*?"
        r"Change from baseline: ([\d.-]+).*?"
        r"Z-score: ([\d.-]+).*?"
        r"Statistically significant change: (True|False)",
        re.S
    )

    for match in pattern.finditer(text_data):
        k_value = match.group(1)
        mean_connectivity = match.group(2)
        change = match.group(3)
        z_score = match.group(4)
        significant = match.group(5)
        rows.append([k_value, mean_connectivity, change, z_score, significant])

    with open(output_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

for constellation in ["kuiper_630", "starlink_550", "telesat_1015"]:
    dir = f"figs/{constellation}/5000ms_for_200s/"
    input_file_path = dir + "z_score_analysis_connectivity.txt"
    output_file_path = dir + "z_score_analysis_connectivity.csv"  # Replace with the desired output path for the CSV file
    text_to_csv(input_file_path, output_file_path)
