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

def analyse_network_resilience(G, k_values=range(0, 251, 25)):
    """
    Analyse network resilience by removing top-k nodes based on betweenness centrality
    and recalculating metrics for each k value.
    """
    try:
        print(f"Starting resilience analysis with {len(G.nodes())} nodes")
        results = {}
        
        # Get initial betweenness centrality
        print("Calculating initial betweenness centrality...")
        betweenness = nx.betweenness_centrality(G)
        
        # Filter satellite nodes and sort by betweenness
        satellite_nodes = [(node, score) for node, score in betweenness.items() 
                          if G.nodes[node].get('type') == 'satellite']
        satellite_nodes.sort(key=lambda x: x[1], reverse=True)
        print(f"Found {len(satellite_nodes)} satellite nodes")
        
        for k in tqdm(k_values, desc="Processing k values"):
            print(f"\nAnalysing network with k={k}")
            # Create a copy of the graph for this iteration
            G_k = G.copy()
            
            # Remove top k satellite nodes
            nodes_to_remove = [node for node, _ in satellite_nodes[:k]]
            G_k.remove_nodes_from(nodes_to_remove)
            print(f"Removed {k} nodes, {len(G_k.nodes())} nodes remaining")
            
            # Calculate metrics for reduced graph
            metrics = analyse_gs_connectivity(G_k, None)
            if metrics:
                results[k] = metrics
                print(f"k={k}: Connectivity: {metrics['connectivity_percentage']:.2f}%")
            else:
                print(f"Failed to calculate metrics for k={k}")
            
        return results
        
    except Exception as e:
        print(f"Error in analyse_network_resilience: {str(e)}")
        traceback.print_exc()
        return None

def save_resilience_metrics(metrics, output_dir):
    """Save resilience analysis metrics to CSV"""
    try:
        print(f"Saving resilience metrics to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        resilience_csv_path = os.path.join(output_dir, 'resilience_metrics.csv')
        
        headers = [
            'timestamp',
            'k_value',
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
        
        # Create new file with headers if it doesn't exist
        if not os.path.exists(resilience_csv_path):
            print("Creating new resilience metrics file")
            with open(resilience_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        
        # Write metrics for each k value
        rows = []
        print(f"Processing {len(metrics)} k values")
        for k, k_metrics in metrics.items():
            row = [
                k_metrics['timestamp'],
                k,
                k_metrics['average_path_length'],
                k_metrics['num_connected_pairs'],
                k_metrics['total_pairs'],
                k_metrics['connectivity_percentage'],
                k_metrics['min_path_length'],
                k_metrics['max_path_length'],
                k_metrics['num_ground_stations'],
                k_metrics['num_satellites'],
                k_metrics['average_hops']
            ]
            rows.append(row)
        
        print(f"Writing {len(rows)} rows to CSV")
        with open(resilience_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        print("Successfully saved resilience metrics")
        return resilience_csv_path
        
    except Exception as e:
        print(f"Error saving resilience metrics: {str(e)}")
        traceback.print_exc()
        return None

def process_time_point(t, epoch, satellites, ground_stations, list_isls, max_gsl_length_m, max_isl_length_m, csv_paths, pickles_path, network):
    try:
        print(f"\nProcessing time point {t}")
        path_csv_path, betweenness_csv_path = csv_paths
        
        # Construct graph
        print("Constructing graph...")
        graph = construct_graph_with_distances(
            epoch, t, satellites, ground_stations,
            list_isls, max_gsl_length_m, max_isl_length_m, network
        )

        print("Saving graph pickle...")
        with open(f"{pickles_path}/{t}.pickle", "wb") as f:
            pickle.dump(graph, f)
        
        if graph is None:
            print("Failed to construct graph")
            return t, False
            
        print(f"Graph constructed with {len(graph.nodes())} nodes")
            
        # Calculate base metrics
        print("Calculating base metrics...")
        metrics = analyse_gs_connectivity(graph, t)
        if metrics is None:
            print("Failed to calculate base metrics")
            return t, False
        
        print(f"Base connectivity: {metrics['connectivity_percentage']:.2f}%")
            
        # Save path metrics
        print("Saving path metrics...")
        success1 = save_timestep_metrics(metrics, path_csv_path)
        
        # Save betweenness metrics
        print("Saving betweenness metrics...")
        success2 = save_betweenness_metrics(t, graph, metrics['betweenness'], betweenness_csv_path)
        
        # Perform resilience analysis
        print("Starting resilience analysis...")
        resilience_metrics = analyse_network_resilience(graph)
        
        if resilience_metrics is None:
            print("Failed to complete resilience analysis")
            return t, False
        
        # Save resilience metrics
        print("Saving resilience metrics...")
        resilience_dir = os.path.dirname(path_csv_path)
        for k, k_metrics in resilience_metrics.items():
            k_metrics['timestamp'] = t  # Add timestamp to metrics
        success3 = save_resilience_metrics(resilience_metrics, resilience_dir)
        
        print(f"Time point {t} processing complete")
        return t, (success1 and success2 and success3)
        
    except Exception as e:
        print(f"Error processing time {t}: {str(e)}")
        traceback.print_exc()
        return t, False

def resilience_analysis(
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