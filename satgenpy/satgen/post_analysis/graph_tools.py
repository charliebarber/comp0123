# The MIT License (MIT)
#
# Copyright (c) 2020 ETH Zurich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from satgen.distance_tools import *
import networkx as nx
from astropy import units as u
import numpy as np

def construct_graph_with_distances(epoch, time_since_epoch_ns, satellites, ground_stations, list_isls,
                                   max_gsl_length_m, max_isl_length_m):
    try:
        # Time calculation
        time = epoch + time_since_epoch_ns * u.ns
        str_epoch = str(epoch)
        str_time = str(time)
        
        # Create graph
        sat_net_graph = nx.Graph()
        n_satellites = len(satellites)
        
        # Add satellite nodes
        for i in range(n_satellites):
            sat_net_graph.add_node(i, type='satellite')
            
        # Add ground station nodes
        for i in range(len(ground_stations)):
            gs_node = n_satellites + i
            sat_net_graph.add_node(gs_node, type='ground_station')
        
        # Process ISLs
        if list_isls:
            sat_pairs = np.array(list_isls)
            distances = [
                distance_m_between_satellites(satellites[a], satellites[b], str_epoch, str_time)
                for a, b in sat_pairs
            ]
            
            valid_isls = np.array(distances) <= max_isl_length_m
            valid_pairs = sat_pairs[valid_isls]
            valid_distances = np.array(distances)[valid_isls]
            
            for (a, b), d in zip(valid_pairs, valid_distances):
                sat_net_graph.add_edge(a, b, weight=d, type='ISL')
        
        # Process ground station connections
        for gid, ground_station in enumerate(ground_stations):
            gs_node = n_satellites + gid
            
            distances = [
                distance_m_ground_station_to_satellite(ground_station, sat, str_epoch, str_time)
                for sat in satellites
            ]
            
            valid_sats = np.where(np.array(distances) <= max_gsl_length_m)[0]
            
            for sat_id in valid_sats:
                sat_net_graph.add_edge(gs_node, sat_id, weight=distances[sat_id], type='GSL')
        
        return sat_net_graph
        
    except Exception as e:
        print(f"Error constructing graph: {str(e)}")
        return None

def get_satellites(G):
    """Return all satellite nodes (partition 0) in the bipartite graph"""
    return {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}

def get_ground_stations(G):
    """Return all ground station nodes (partition 1) in the bipartite graph"""
    return {n for n, d in G.nodes(data=True) if d['bipartite'] == 1}

def compute_path_length_with_graph(path, graph):
    return sum_path_weights(augment_path_with_weights(path, graph))


def compute_path_length_without_graph(path, epoch, time_since_epoch_ns, satellites, ground_stations, list_isls,
                                      max_gsl_length_m, max_isl_length_m):

    # Time
    time = epoch + time_since_epoch_ns * u.ns

    # Go hop-by-hop and compute
    path_length_m = 0.0
    for i in range(1, len(path)):
        
        from_node_id = path[i - 1]
        to_node_id = path[i]
        
        # Satellite to satellite
        if from_node_id < len(satellites) and to_node_id < len(satellites):
            sat_distance_m = distance_m_between_satellites(
                satellites[from_node_id],
                satellites[to_node_id],
                str(epoch),
                str(time)
            )
            if sat_distance_m > max_isl_length_m \
                    or ((to_node_id, from_node_id) not in list_isls and (from_node_id, to_node_id) not in list_isls):
                raise ValueError("Invalid ISL hop")
            path_length_m += sat_distance_m

        # Ground station to satellite
        elif from_node_id >= len(satellites) and to_node_id < len(satellites):
            ground_station = ground_stations[from_node_id - len(satellites)]
            distance_m = distance_m_ground_station_to_satellite(
                ground_station,
                satellites[to_node_id],
                str(epoch),
                str(time)
            )
            if distance_m > max_gsl_length_m:
                raise ValueError("Invalid GSL hop from " + str(from_node_id) + " to " + str(to_node_id)
                                 + " (" + str(distance_m) + " larger than " + str(max_gsl_length_m) + ")")
            path_length_m += distance_m

        # Satellite to ground station
        elif from_node_id < len(satellites) and to_node_id >= len(satellites):
            ground_station = ground_stations[to_node_id - len(satellites)]
            distance_m = distance_m_ground_station_to_satellite(
                ground_station,
                satellites[from_node_id],
                str(epoch),
                str(time)
            )
            if distance_m > max_gsl_length_m:
                raise ValueError("Invalid GSL hop from " + str(from_node_id) + " to " + str(to_node_id)
                                 + " (" + str(distance_m) + " larger than " + str(max_gsl_length_m) + ")")
            path_length_m += distance_m

        else:
            raise ValueError("Hops between ground stations are not permitted: %d -> %d" % (from_node_id, to_node_id))

    return path_length_m


def get_path(src, dst, forward_state):

    if forward_state[(src, dst)] == -1:  # No path exists
        return None

    curr = src
    path = [src]
    while curr != dst:
        next_hop = forward_state[(curr, dst)]
        path.append(next_hop)
        curr = next_hop
    return path


def get_path_with_weights(src, dst, forward_state, sat_net_graph_with_gs):

    if forward_state[(src, dst)] == -1:  # No path exists
        return None

    curr = src
    path = []
    while curr != dst:
        next_hop = forward_state[(curr, dst)]
        w = sat_net_graph_with_gs.get_edge_data(curr, next_hop)["weight"]
        path.append((curr, next_hop, w))
        curr = next_hop
    return path


def augment_path_with_weights(path, sat_net_graph_with_gs):
    res = []
    for i in range(1, len(path)):
        res.append((path[i - 1], path[i], sat_net_graph_with_gs.get_edge_data(path[i - 1], path[i])["weight"]))
    return res


def sum_path_weights(weighted_path):
    res = 0.0
    for i in weighted_path:
        res += i[2]
    return res
