import dimod
import networkx as nx
import time
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import dimod
from ortools.linear_solver import pywraplp

# token = "DEV-7affe1a83dbe06fa17c9a260577608396c251455"
# token = "DEV-b28f5c26b9419829978caa8899867ab5c25f9802"

# DEV-898779584c4bed23fcf5bcbd657344d29493c2b9
#pip install dwave-ocean-sdk
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.samplers import SimulatedAnnealingSampler
from itertools import combinations

def create_mis_qubo(graph, penalty_weight=1.0):
    """
    Create a QUBO formulation for the Maximum Independent Set problem.

    Parameters:
        graph (networkx.Graph): Input graph with nodes and edges.
        penalty_weight (float): Weight for the penalty term.

    Returns:
        dimod.BinaryQuadraticModel: The QUBO for the Maximum Independent Set.
    """
    bqm = dimod.BinaryQuadraticModel('BINARY')

    # Add linear terms for maximizing the size of the independent set
    for node in graph.nodes:
        bqm.add_variable(node, -1)  # Coefficient for H1: -x_i

    # Add quadratic terms for penalizing adjacent nodes in the set
    for edge in graph.edges:
        i, j = edge
        bqm.add_interaction(i, j, penalty_weight)  # Coefficient for H2: lambda * x_i * x_j

    return bqm

if __name__ == "__main__":
    input_folder = "input_data"  # Thư mục chứa các file TXT
    file_to_read = "data_4.txt"  # File cần đọc

    # Đường dẫn đầy đủ đến file
    file_path = os.path.join(input_folder, file_to_read)

    # Kiểm tra nếu file tồn tại
    if os.path.exists(file_path):
        # Tạo đồ thị từ file
        G = nx.Graph()
        with open(file_path, "r") as file:
            for line in file:
                u, v = map(int, line.split())  # Đọc các cạnh từ file
                G.add_edge(u, v)
    # print(G.edges)
                
    # # visualize the graph and save to folder image
    # # output_folder = "image"
    # # if not os.path.exists(output_folder):
    # #     os.makedirs(output_folder)
    
    # # plt.figure(figsize=(8, 6))
    # # nx.draw(G, with_labels=True, node_color='skyblue', node_size=3000, font_size=10, font_weight='bold', edge_color='gray')

    # # output_file_path = os.path.join(output_folder, "map_10.png")
    # # plt.savefig(output_file_path)
    # # print(f"Đã lưu đồ thị vào {output_file_path}")
    
    # using ortool
    # list_of_ones = [1] * len(G.nodes)
    # res_ortools = maximum_weighted_independent_set(list_of_ones, G.edges())
    #print("Ortool: ", res_ortools)
    
    # Generate QUBO
    penalty_weigth_num = 0.5
    Q = create_mis_qubo(G, penalty_weight=penalty_weigth_num)
    print(Q)
    
    # Quantum Annealing
    # chainstrength = 8
    # numruns = 1000
    # annealingTime = 50
    # sampler = EmbeddingComposite(DWaveSampler(token='DEV-898779584c4bed23fcf5bcbd657344d29493c2b9'))
    # response = sampler.sample(Q,
    #                            chain_strength=chainstrength,
    #                            num_reads=numruns,
    #                            annealing_time=annealingTime,
    #                            label='Maximum Independent Set')
    
    # Simulated Annealing
    ##sampleset = dimod.SimulatedAnnealingSampler()
    # Measure time for Simulated Annealing
    sampler = SimulatedAnnealingSampler()

    response = sampler.sample(Q, num_reads = 1000)
    
    # print(response['time'])
    print(response.info)