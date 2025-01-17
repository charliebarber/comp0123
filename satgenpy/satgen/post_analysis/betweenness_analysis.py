from .graph_tools import *
from satgen.distance_tools import *
from satgen.isls import *
from satgen.ground_stations import *
from satgen.tles import *
import exputil
import numpy as np
from .print_routes_and_rtt import print_routes_and_rtt
from statsmodels.distributions.empirical_distribution import ECDF
import csv


SPEED_OF_LIGHT_M_PER_S = 299792458.0

GEODESIC_ECDF_PLOT_CUTOFF_KM = 500

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
from tqdm.auto import tqdm
import time
import json
import os
import networkx as nx
import pandas as pd

import pickle


class NetworkpathsEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types and network data"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): self.default(v) for k, v in obj.items()}
        return super().default(obj)

def convert_to_serializable(data):
    """Convert a dictionary with NumPy values to JSON-serializable format"""
    if isinstance(data, dict):
        return {str(k): convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    return data

def initialise_metrics_csv(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Path metrics CSV
    path_csv_path = os.path.join(output_dir, 'path_metrics.csv')
    path_headers = [
        'timestamp',
        'average_path_length',
        'num_connected_pairs',
        'total_pairs',
        'connectivity_percentage',
        'min_path_length',
        'max_path_length',
        'num_ground_stations',
        'num_satellites',
        'average_hops'
    ]
    
    with open(path_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(path_headers)
    
    # Betweenness CSV
    betweenness_csv_path = os.path.join(output_dir, 'betweenness_metrics.csv')
    betweenness_headers = ['timestamp', 'node_id', 'node_type', 'betweenness_centrality']
    
    with open(betweenness_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(betweenness_headers)
    
    return path_csv_path, betweenness_csv_path

def save_betweenness_metrics(timestamp, G, betweenness, csv_path):
    """Save betweenness metrics for a timestep"""
    if betweenness is None:
        return False
        
    try:
        rows = []
        for node, score in betweenness.items():
            node_type = G.nodes[node].get('type', 'unknown')
            rows.append([timestamp, node, node_type, score])
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        return True
    except Exception as e:
        print(f"Error saving betweenness metrics: {str(e)}")
        return False

def save_timestep_metrics(metrics, csv_path):
    """Save metrics for a single timestep to CSV file"""
    if metrics is None:
        return False
        
    try:
        row = [
            metrics['timestamp'],
            metrics['average_path_length'],
            metrics['num_connected_pairs'],
            metrics['total_pairs'],
            metrics['connectivity_percentage'],
            metrics['min_path_length'],
            metrics['max_path_length'],
            metrics['num_ground_stations'],
            metrics['num_satellites'],
            metrics['average_hops']
        ]
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        return True
    except Exception as e:
        print(f"Error saving metrics to CSV: {str(e)}")
        return False

def calculate_gs_path_betweenness(G, paths):
    try:
        # Initialize betweenness scores for all nodes
        betweenness = {node: 0.0 for node in G.nodes()}
        
        # Get number of valid paths for denominator
        valid_paths = sum(1 for path in paths.values() if path is not None)
        if valid_paths == 0:
            return betweenness
        
        # Count how many times each node appears in valid paths
        for path in paths.values():
            if path is None:
                continue
            
            # For each node in the path (excluding endpoints)
            for node in path[1:-1]:
                betweenness[node] += 1.0
        
        # Normalize by number of valid paths to get percentage of paths node appears in
        for node in betweenness:
            betweenness[node] /= valid_paths
        
        return betweenness
        
    except Exception as e:
        print(f"Error calculating betweenness: {str(e)}")
        return None
        
    except Exception as e:
        print(f"Error calculating betweenness: {str(e)}")
        return None

def calculate_gs_paths(G):
    """Calculate paths between ground stations"""
    try:
        # Get ground station nodes
        ground_stations = [n for n, d in G.nodes(data=True) if d.get('type') == 'ground_station']
        if not ground_stations:
            raise ValueError("No ground stations found")
        
        # initialise storage
        path_lengths = {}
        paths = {}
        valid_paths = 0
        total_length = 0
        
        # Calculate shortest path between each GS pair
        for i, source in enumerate(ground_stations):
            for dest in ground_stations[i+1:]:
                try:
                    path = nx.shortest_path(G, source, dest, weight='weight')
                    path_length = sum(G[path[i]][path[i+1]]['weight'] 
                                    for i in range(len(path)-1))
                    
                    path_lengths[(source, dest)] = path_length
                    paths[(source, dest)] = path
                    valid_paths += 1
                    total_length += path_length
                    
                except nx.NetworkXNoPath:
                    path_lengths[(source, dest)] = float('inf')
                    paths[(source, dest)] = None
        
        avg_path_length = total_length / valid_paths if valid_paths > 0 else float('inf')
        return avg_path_length, path_lengths, paths
        
    except Exception as e:
        print(f"Error in calculate_gs_paths: {str(e)}")
        return None, None, None

def analyse_gs_connectivity(G, t):
    try:
        avg_length, lengths, paths = calculate_gs_paths(G)
        if avg_length is None:
            return None
            
        # Calculate betweenness based on GS paths
        betweenness = calculate_gs_path_betweenness(G, paths)
        
        # Get node sets
        ground_stations = [n for n, d in G.nodes(data=True) if d.get('type') == 'ground_station']
        satellites = [n for n, d in G.nodes(data=True) if d.get('type') == 'satellite']
        
        # Calculate metrics
        metrics = {
            'timestamp': t,
            'average_path_length': avg_length,
            'num_connected_pairs': sum(1 for l in lengths.values() if l != float('inf')),
            'total_pairs': len(lengths),
            'min_path_length': min((l for l in lengths.values() if l != float('inf')), default=float('inf')),
            'max_path_length': max((l for l in lengths.values() if l != float('inf')), default=float('inf')),
            'num_ground_stations': len(ground_stations),
            'num_satellites': len(satellites),
            'average_hops': sum(len(p)-1 for p in paths.values() if p is not None) / 
                           sum(1 for p in paths.values() if p is not None) if any(p is not None for p in paths.values()) else float('inf'),
            'betweenness': betweenness  # Add betweenness to metrics
        }
        
        metrics['connectivity_percentage'] = (metrics['num_connected_pairs'] / metrics['total_pairs'] * 100 
                                            if metrics['total_pairs'] > 0 else 0)
        
        return metrics
        
    except Exception as e:
        print(f"Error in analyse_gs_connectivity: {str(e)}")
        return None

def process_time_point(t, epoch, satellites, ground_stations, list_isls, max_gsl_length_m, max_isl_length_m, csv_paths, pickles_path, network):
    try:
        path_csv_path, betweenness_csv_path = csv_paths
        
        # Construct graph
        graph = construct_graph_with_distances(
            epoch, t, satellites, ground_stations,
            list_isls, max_gsl_length_m, max_isl_length_m, network
        )

        with open(f"{pickles_path}/{t}.pickle", "wb") as f:
            pickle.dump(graph, f)
        
        if graph is None:
            return t, False
            
        # Calculate metrics
        metrics = analyse_gs_connectivity(graph, t)
        if metrics is None:
            return t, False
            
        # Save path metrics
        success1 = save_timestep_metrics(metrics, path_csv_path)
        
        # Save betweenness metrics
        success2 = save_betweenness_metrics(t, graph, metrics['betweenness'], betweenness_csv_path)
        
        return t, (success1 and success2)
        
    except Exception as e:
        print(f"Error processing time {t}: {str(e)}")
        return t, False

def betweenness_analysis(
        output_data_dir, satellite_network_dir, dynamic_state_update_interval_ms,
        simulation_end_time_s, satgenpy_dir_with_ending_slash
):
    print("Starting path analysis...")
    
    core_network_folder_name = satellite_network_dir.split("/")[-1]
    print(f"core_network_folder_name: {core_network_folder_name}")
    base_output_dir = "%s/%s/%dms_for_%ds" % (
        output_data_dir, core_network_folder_name, dynamic_state_update_interval_ms, simulation_end_time_s
    )
    
    # Load all data
    print("Loading input data...")
    ground_stations = read_ground_stations_extended(satellite_network_dir + "/ground_stations.txt")
    tles = read_tles(satellite_network_dir + "/tles.txt")
    satellites = tles["satellites"]
    list_isls = read_isls(satellite_network_dir + "/isls.txt", len(satellites))
    epoch = tles["epoch"]
    description = exputil.PropertiesConfig(satellite_network_dir + "/description.txt")

    # Constants
    print("Setting up parameters...")
    simulation_end_time_ns = simulation_end_time_s * 1000 * 1000 * 1000
    dynamic_state_update_interval_ns = dynamic_state_update_interval_ms * 1000 * 1000
    max_gsl_length_m = exputil.parse_positive_float(description.get_property_or_fail("max_gsl_length_m"))
    max_isl_length_m = exputil.parse_positive_float(description.get_property_or_fail("max_isl_length_m"))

    paths_dir = os.path.join(base_output_dir, 'network_paths')
    path_csv_path, betweenness_csv_path = initialise_metrics_csv(paths_dir)
    pickles_path = os.path.join(base_output_dir, 'pickles')
    os.makedirs(pickles_path, exist_ok=True)

    # Generate time points
    time_points = list(range(0, simulation_end_time_ns, dynamic_state_update_interval_ns))
    num_time_points = len(time_points)
    print(f"Will process {num_time_points} time points")
    
    # Create thread pool
    num_workers = min(8, len(time_points))
    print(f"Using {num_workers} worker threads")
    
    # Track completion
    completed = 0
    successful = 0
    
    print("\nStarting parallel processing...")
    
    # Process with progress bar and batches
    with tqdm(total=num_time_points, desc="Processing time points") as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Process in batches
            batch_size = 10
            for i in range(0, len(time_points), batch_size):
                batch = time_points[i:i + batch_size]
                print(f"\nProcessing batch starting at time point {batch[0]}")
                
                # Submit batch of jobs
                future_to_time = {
                    executor.submit(
                        process_time_point, 
                        t, 
                        epoch, 
                        satellites, 
                        ground_stations, 
                        list_isls, 
                        max_gsl_length_m, 
                        max_isl_length_m,
                        (path_csv_path, betweenness_csv_path),
                        pickles_path,
                        core_network_folder_name
                    ): t for t in batch
                }
                
                # Process batch results
                for future in concurrent.futures.as_completed(future_to_time):
                    t = future_to_time[future]
                    try:
                        result = future.result()
                        if result[1]:  # If processing was successful
                            successful += 1
                        completed += 1
                        pbar.update(1)
                        print(f"Completed {completed}/{num_time_points} time points (Successful: {successful})")
                    except Exception as e:
                        print(f"\nError processing time {t}: {str(e)}")
                        pbar.update(1)
    
    print("\nFinished parallel processing")
    print(f"Successfully processed {successful}/{num_time_points} time points")
    print(f"Metrics saved to {path_csv_path}, {betweenness_csv_path}")
    
    print("Network analysis complete")
    return successful