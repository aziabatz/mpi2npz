import numpy as np
import json
import glob
import os
import re

def init_matrix(P, edges, volume, n_msgs, reward_type="volume", normalize=False):
    adj_matrix = np.zeros((P, P))
    if reward_type == "volume":
        rewards = volume
    elif reward_type == "n_msgs":
        rewards = n_msgs
    elif reward_type == "num_msgs":
        rewards = n_msgs
    else:
        raise ValueError("Unsupported reward type")
    
    for edge, cost in zip(edges, rewards):
        node1, node2 = edge
        adj_matrix[node1][node2] = cost

    if normalize:
        max_cost = np.max(adj_matrix)
        if max_cost > 0:
            adj_matrix = adj_matrix / max_cost

    return adj_matrix

def process_json_file(file_path, reward_type="volume", normalize=False):
    adj_matrix = None
    optimal = None
    with open(file_path, 'r') as f:
        data = json.load(f)

        P = data["Graph"]["P"]
        edges = data["Graph"]["comms"]["edges"]
        volume = data["Graph"]["comms"]["volume"]
        n_msgs = data["Graph"]["comms"]["n_msgs"]

        adj_matrix = init_matrix(P, edges, volume, n_msgs, reward_type, normalize)
        optimal = data.get("Benchmark", {}).get("optimal_mapping", np.zeros(shape=(P,)))

    return adj_matrix, optimal


def by_proc_mach(root_dir=".", reward_type="volume", normalize=False):
    
    # Subdir recursive walk
    for subdir, dirs, files in os.walk(root_dir):
        # Key is PROCS_MACHINES
        config_group = {}
        
        for file in files:
            if file.endswith('.json'):
                # Regex Hell
                match = re.match(r'.*?(\d+_\d+).*?\.json', file)
                if match:
                    key = match.group(1)
                    if key not in config_group:
                        config_group[key] = {"matrices": [], "optimals": []}
                    
                    file_path = os.path.join(subdir, file)

                    adj_matrix, optimal = process_json_file(file_path=file_path)

                    config_group[key]["matrices"].append(adj_matrix)
                    config_group[key]["optimals"].append(optimal)

        for key, value in config_group.items():
            stacked_matrices = np.stack(value["matrices"], axis=0, dtype=np.float32)
            print(key, value['optimals'])
            stacked_optimals = np.stack(value["optimals"], axis=0, dtype=np.int32)

            np.savez(os.path.join(subdir, f"{key}.npz"), cost_matrix=stacked_matrices, optimals=stacked_optimals)


by_proc_mach()
