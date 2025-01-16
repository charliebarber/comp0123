from .graph_tools import *
from satgen.distance_tools import *
from satgen.isls import *
from satgen.ground_stations import *
from satgen.tles import *
import exputil
import numpy as np
from .print_routes_and_rtt import print_routes_and_rtt
from statsmodels.distributions.empirical_distribution import ECDF


SPEED_OF_LIGHT_M_PER_S = 299792458.0

GEODESIC_ECDF_PLOT_CUTOFF_KM = 500

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm.auto import tqdm
import time
import json
import os
import networkx as nx
import pandas as pd


class NetworkPropertiesEncoder(json.JSONEncoder):
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

def calculate_degree_distribution(G):
    """Calculate the frequency distribution of degrees in the network"""
    degrees = dict(G.degree())
    degree_freq = {}
    for degree in degrees.values():
        degree_freq[degree] = degree_freq.get(degree, 0) + 1
    return degree_freq

def calculate_network_properties(G, time_since_epoch_ns):
    """Calculate network properties for a given graph"""
    try:
        # Calculate degree distribution (frequency of each degree)
        degree_freq = calculate_degree_distribution(G)
        
        properties = {
            'timestamp': int(time_since_epoch_ns),
            
            # Degree properties
            'degree_distribution': {str(k): int(v) for k, v in degree_freq.items()},  # How many nodes have each degree
            'avg_degree': float(sum(dict(G.degree()).values()) / G.number_of_nodes()),
            
            # Path length properties
            'avg_shortest_path': float(nx.average_shortest_path_length(G, weight='weight')) if nx.is_connected(G) else None,
            
            # Betweenness centrality
            'betweenness_centrality': {str(k): float(v) for k, v in nx.betweenness_centrality(G, weight='weight').items()},
            
            # Basic properties
            'num_nodes': int(G.number_of_nodes()),
            'num_edges': int(G.number_of_edges()),
            
            # Separate ISL and GSL stats
            'num_isls': int(len([(u,v) for (u,v,d) in G.edges(data=True) if d.get('type')=='ISL'])),
            'num_gsls': int(len([(u,v) for (u,v,d) in G.edges(data=True) if d.get('type')=='GSL']))
        }
        return properties
    except Exception as e:
        print(f"Error calculating properties for timestamp {time_since_epoch_ns}: {str(e)}")
        return None

def save_properties_to_files(all_properties, output_dir):
    """Save network properties to various file formats"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert properties to serializable format
    serializable_properties = [convert_to_serializable(props) for props in all_properties if props is not None]
    
    # Save full data as JSON using custom encoder
    with open(os.path.join(output_dir, 'network_properties.json'), 'w') as f:
        json.dump(serializable_properties, f, cls=NetworkPropertiesEncoder, indent=2)
    
    # Create time series of scalar metrics
    time_series_data = []
    for props in all_properties:
        if props is None:
            continue
        
        scalar_metrics = {
            'timestamp': int(props['timestamp']),
            'avg_degree': float(props['avg_degree']),
            'avg_shortest_path': float(props['avg_shortest_path']) if props['avg_shortest_path'] is not None else None,
            'num_nodes': int(props['num_nodes']),
            'num_edges': int(props['num_edges']),
            'num_isls': int(props['num_isls']),
            'num_gsls': int(props['num_gsls'])
        }
        time_series_data.append(scalar_metrics)
    
    # Save time series as CSV
    df = pd.DataFrame(time_series_data)
    df.to_csv(os.path.join(output_dir, 'network_metrics_time_series.csv'), index=False)
    
    # Save degree distributions
    degree_distributions = []
    for props in all_properties:
        if props is None:
            continue
            
        timestamp = props['timestamp']
        degree_dist = props['degree_distribution']
        
        # For each degree value and its frequency
        for degree, frequency in degree_dist.items():
            degree_distributions.append({
                'timestamp': timestamp,
                'degree': int(degree),
                'frequency': int(frequency),
                'frequency_normalized': float(frequency) / props['num_nodes']  # Add normalized frequency
            })
    
    # Save degree distributions as CSV
    df_degrees = pd.DataFrame(degree_distributions)
    df_degrees.to_csv(os.path.join(output_dir, 'degree_distributions.csv'), index=False)

def network_analysis(
        output_data_dir, satellite_network_dir, dynamic_state_update_interval_ms,
        simulation_end_time_s, satgenpy_dir_with_ending_slash
):
    print("Starting network analysis...")
    
    core_network_folder_name = satellite_network_dir.split("/")[-1]
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

    # Generate time points
    time_points = list(range(0, simulation_end_time_ns, dynamic_state_update_interval_ns))
    num_time_points = len(time_points)
    print(f"Will process {num_time_points} time points")
    
    # Create a thread pool with fewer workers
    num_workers = min(8, len(time_points))
    print(f"Using {num_workers} worker threads")
    
    # Store results
    results = {}
    network_properties = []
    completed = 0
    
    def process_time_point(t):
        """Process a single time point"""
        try:
            print(f"\nStarting time point {t}")
            start_time = time.time()
            
            # Construct graph with edge types
            graph = construct_graph_with_distances(
                epoch, t, satellites, ground_stations,
                list_isls, max_gsl_length_m, max_isl_length_m
            )
            
            # Add edge types to the graph
            for u, v, d in graph.edges(data=True):
                if u < len(satellites) and v < len(satellites):
                    d['type'] = 'ISL'
                else:
                    d['type'] = 'GSL'
            
            # Calculate properties
            properties = calculate_network_properties(graph, t)
            
            end_time = time.time()
            print(f"Completed time point {t} in {end_time - start_time:.2f} seconds")
            return t, graph, properties
            
        except Exception as e:
            print(f"\nError processing time {t}: {str(e)}")
            return t, None, None
    
    print("\nStarting parallel processing...")
    
    # Process with progress bar and smaller batches
    with tqdm(total=num_time_points, desc="Processing time points") as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Process in smaller batches
            batch_size = 10
            for i in range(0, len(time_points), batch_size):
                batch = time_points[i:i + batch_size]
                print(f"\nProcessing batch starting at time point {batch[0]}")
                
                # Submit batch of jobs
                future_to_time = {
                    executor.submit(process_time_point, t): t 
                    for t in batch
                }
                
                # Process batch results
                for future in concurrent.futures.as_completed(future_to_time):
                    t = future_to_time[future]
                    try:
                        result = future.result()
                        if result[1] is not None:  # If graph was created successfully
                            results[result[0]] = result[1]
                            network_properties.append(result[2])
                        completed += 1
                        pbar.update(1)
                        print(f"Completed {completed}/{num_time_points} time points")
                    except Exception as e:
                        print(f"\nError processing time {t}: {str(e)}")
                        pbar.update(1)
    
    print("\nFinished parallel processing")
    print(f"Successfully processed {len(results)} time points")
    
    # Save network properties
    properties_dir = os.path.join(base_output_dir, 'network_properties')
    save_properties_to_files(network_properties, properties_dir)
    print(f"Saved network properties to {properties_dir}")
    
    # Sort results by time
    sorted_results = [results[t] for t in sorted(results.keys()) if results[t] is not None]
    
    print("Network analysis complete")
    return sorted_results