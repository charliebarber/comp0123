import networkx as nx
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from tqdm import tqdm
import signal
import sys
from multiprocessing import cpu_count
import time
import datetime

def signal_handler(signum, frame):
    """Handle interrupt signals."""
    print("\nReceived interrupt signal. Cleaning up...")
    sys.exit(1)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Analysis functions
def load_graph(pickle_path: str) -> nx.Graph:
    """Load a networkx graph from a pickle file."""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def load_centrality_data(csv_path: str) -> pd.DataFrame:
    """Load and process betweenness centrality data from CSV."""
    return pd.read_csv(csv_path)

def get_timestamps_from_pickles(pickle_dir: str) -> List[int]:
    """Get sorted list of timestamps from pickle filenames."""
    pickle_files = glob.glob(os.path.join(pickle_dir, "*.pickle"))
    timestamps = [int(os.path.splitext(os.path.basename(f))[0]) for f in pickle_files]
    return sorted(timestamps)

def get_ground_stations(G: nx.Graph) -> List[int]:
    """Get ground station nodes."""
    return [n for n, d in G.nodes(data=True) if d.get('type') == 'ground_station']

def get_satellites(G: nx.Graph) -> List[int]:
    """Get satellite nodes."""
    return [n for n, d in G.nodes(data=True) if d.get('type') == 'satellite']

def get_top_k_satellite_nodes(centrality_data: pd.DataFrame, 
                            timestamp: int, 
                            k: int) -> List[int]:
    """Get the top k satellite nodes by betweenness centrality."""
    filtered_data = centrality_data[
        (centrality_data['timestamp'] == timestamp) & 
        (centrality_data['node_type'] == 'satellite')
    ]
    top_k = filtered_data.nlargest(k, 'betweenness_centrality')
    return top_k['node_id'].tolist()

def calculate_gs_paths(G: nx.Graph, ground_stations: List[int]) -> Tuple[float, Dict, Dict, List]:
    """Calculate paths between ground stations."""
    try:
        path_lengths = {}
        paths = {}
        valid_paths = 0
        total_length = 0
        disconnected_pairs = []
        
        if not ground_stations:
            return float('inf'), {}, {}, []
        
        gs_pairs = list(itertools.combinations(ground_stations, 2))
        
        for source, dest in gs_pairs:
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
                disconnected_pairs.append((source, dest))
        
        avg_path_length = total_length / valid_paths if valid_paths > 0 else float('inf')
        return avg_path_length, path_lengths, paths, disconnected_pairs
        
    except Exception as e:
        print(f"Warning in calculate_gs_paths: {str(e)}")
        return float('inf'), {}, {}, []

def calculate_network_metrics(G: nx.Graph) -> Dict:
    """Calculate network metrics focusing on ground station connectivity."""
    metrics = {}
    
    ground_stations = get_ground_stations(G)
    satellites = get_satellites(G)
    
    avg_length, lengths, paths, disconnected_pairs = calculate_gs_paths(G, ground_stations)
    
    connected_paths = sum(1 for l in lengths.values() if l != float('inf'))
    total_paths = len(lengths)
    
    metrics.update({
        'avg_path_length': avg_length,
        'num_connected_pairs': connected_paths,
        'num_disconnected_pairs': len(disconnected_pairs),
        'total_pairs': total_paths,
        'min_path_length': min((l for l in lengths.values() if l != float('inf')), default=float('inf')),
        'max_path_length': max((l for l in lengths.values() if l != float('inf')), default=float('inf')),
        'num_ground_stations': len(ground_stations),
        'num_satellites': len(satellites),
        'average_hops': sum(len(p)-1 for p in paths.values() if p is not None) / connected_paths if connected_paths > 0 else float('inf'),
        'connectivity_percentage': (connected_paths / total_paths * 100) if total_paths > 0 else 0,
        'disconnected_pairs': disconnected_pairs
    })
    
    return metrics

def process_timestamp_and_k(args: Tuple) -> Dict:
    """Process a single timestamp and k value combination."""
    pickle_dir, centrality_data, timestamp, k = args
    graph_path = os.path.join(pickle_dir, f"{timestamp}.pickle")
    
    try:
        G_original = load_graph(graph_path)
        base_metrics = calculate_network_metrics(G_original)
        
        nodes_to_remove = get_top_k_satellite_nodes(centrality_data, timestamp, k)
        
        G_modified = G_original.copy()
        G_modified.remove_nodes_from(nodes_to_remove)
        
        metrics = calculate_network_metrics(G_modified)
        
        constellation = os.path.basename(os.path.dirname(os.path.dirname(pickle_dir)))
        result = {
            'constellation': constellation,
            'timestamp': timestamp,
            'k': k,
            'nodes_removed': nodes_to_remove,
            'num_connected_pairs': metrics['num_connected_pairs'],
            'num_disconnected_pairs': metrics['num_disconnected_pairs'],
            'connectivity_percentage': metrics['connectivity_percentage'],
            'avg_path_length': metrics['avg_path_length'],
            'min_path_length': metrics['min_path_length'],
            'max_path_length': metrics['max_path_length'],
            'average_hops': metrics['average_hops'],
            'base_connectivity_percentage': base_metrics['connectivity_percentage'],
            'connectivity_percentage_change': metrics['connectivity_percentage'] - base_metrics['connectivity_percentage'],
            'path_length_change': metrics['avg_path_length'] - base_metrics['avg_path_length'],
            'disconnection_change': metrics['num_disconnected_pairs'] - base_metrics['num_disconnected_pairs']
        }
        
        return result
    
    except Exception as e:
        print(f"Error processing timestamp {timestamp} with k={k}: {str(e)}")
        return None

def analyse_constellation(base_dir: str, 
                        constellation: str, 
                        k_values: List[int],
                        max_workers: int,
                        chunk_size: int,
                        pbar: tqdm = None) -> pd.DataFrame:
    """Analyse a single constellation across all timestamps."""
    
    constellation_dir = os.path.join(base_dir, constellation, "2000ms_for_200s")
    pickle_dir = os.path.join(constellation_dir, "pickles")
    csv_path = os.path.join(constellation_dir, "network_paths", "betweenness_metrics.csv")
    
    # Load data once
    centrality_data = load_centrality_data(csv_path)
    timestamps = get_timestamps_from_pickles(pickle_dir)
    
    print(f"\nProcessing {constellation}:")
    print(f"Total timestamps: {len(timestamps)}")
    print(f"Total k values: {len(k_values)}")
    print(f"Total combinations: {len(timestamps) * len(k_values)}")
    print(f"Using {max_workers} threads")
    
    # Create all task combinations
    all_args = [(pickle_dir, centrality_data, t, k) 
                for t, k in itertools.product(timestamps, k_values)]
    
    results = []
    start_time = time.time()
    last_print = start_time
    tasks_completed = 0
    
    # Use a single thread pool for all tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        try:
            # Submit all tasks at once
            future_to_args = {executor.submit(process_timestamp_and_k, args): args 
                            for args in all_args}
            
            print(f"Submitted {len(future_to_args)} tasks to thread pool")
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_args), 
                             total=len(future_to_args),
                             desc=f"Processing {constellation}",
                             leave=False):
                args = future_to_args[future]
                try:
                    result = future.result()
                    if result is not None:
                        with results_lock:
                            results.append(result)
                            tasks_completed += 1
                    
                    # Update progress
                    if pbar:
                        pbar.update(1)
                    
                    # Print performance stats every 60 seconds
                    current_time = time.time()
                    if current_time - last_print >= 60:
                        elapsed = current_time - start_time
                        rate = tasks_completed / elapsed
                        remaining = (len(all_args) - tasks_completed) / rate if rate > 0 else 0
                        print(f"\nPerformance update:")
                        print(f"Tasks completed: {tasks_completed}/{len(all_args)}")
                        print(f"Processing rate: {rate:.2f} tasks/second")
                        print(f"Estimated time remaining: {datetime.timedelta(seconds=int(remaining))}")
                        last_print = current_time
                    
                except Exception as e:
                    print(f"Error processing timestamp {args[2]} with k={args[3]}: {str(e)}")
                    if pbar:
                        pbar.update(1)
        finally:
            print("Finished analyse_constellation")
    
    return pd.DataFrame(results)

def main():
    try:
        base_dir = "data"
        constellations = [
            "kuiper_630",
            "starlink_550",
            "telesat_1015"
        ]
        k_values = list(range(25, 251, 25))  # [25, 50, 75, ..., 225, 250]
        
        # Debug thread calculation
        num_cpus = cpu_count()
        max_workers = max(1, num_cpus - 1)
        print(f"Debug info:")
        print(f"Number of CPUs: {num_cpus}")
        print(f"Initial max_workers: {max_workers}")
        print(f"k_values: {k_values}")
        print(f"len(k_values): {len(k_values)}")
        print(f"len(constellations): {len(constellations)}")
        print(f"(len(k_values) * len(constellations)) // 2: {(len(k_values) * len(constellations)) // 2}")
        
        # Original calculation for comparison
        old_max_workers = min(32, (len(k_values) * len(constellations)) // 2)
        print(f"Old max_workers calculation: {old_max_workers}")
        
        # Use whichever is larger
        max_workers = max(old_max_workers, max_workers)
        chunk_size = max_workers * 4
        print(f"\nFinal configuration:")
        print(f"Using {max_workers} worker threads with chunk size {chunk_size}")
        
        timestamps_sample = get_timestamps_from_pickles(
            os.path.join(base_dir, constellations[0], "2000ms_for_200s", "pickles"))
        total_operations = len(constellations) * len(timestamps_sample) * len(k_values)
        
        all_results = []
        with tqdm(total=total_operations, desc="Overall Progress") as pbar:
            for constellation in constellations:
                print(f"\nAnalysing {constellation}...")
                results = analyse_constellation(
                    base_dir, constellation, k_values, max_workers, chunk_size, pbar)
                all_results.append(results)
                
                results.to_csv(f"results_{constellation}.csv", index=False)
                print(f"Saved results for {constellation}")
        
        final_results = pd.concat(all_results, ignore_index=True)
        final_results.to_csv("results_all_constellations.csv", index=False)
        print("\nAnalysis complete. Results saved to CSV files.")
        
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Cleaning up...")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    finally:
        print("main finished")

if __name__ == "__main__":
    main()