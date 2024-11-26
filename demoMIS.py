import dimod
import networkx as nx
import time
import os
import json
import numpy as np

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

def count_num_penalty(response_data, graph):
    total_penalty = 0
    for data in response_data:
        #print(data)
        res = data.sample
        #print(res, end=" ")
        test_list = []  # khoi tao danh sach cac dinh duoc chon
        penalty = 0  # khoi tao so lan vi pham rang buoc

        # tao list cac dinh duoc chon
        for i in res.keys():
            #print(res.get(i), end=" ")
            if res.get(i) == 1:
                 test_list.append(i)
        #print(test_list, end=" ")

        # tu list do tao nen cac cap canh kha thi, dem so cap thuoc do thi G
        possible_edge = list(combinations(test_list, 2))
        for x in possible_edge:
            if (x in G.edges):
                penalty += 1
                total_penalty += 1
                #print(x, " vi pham rang buoc", end=" ")
            #print(x, end= " ")
        #print("penalty of a row: ", penalty, end=" ")
    #print("\n", "Total penalty of data: ", total_penalty, "\n")
    return total_penalty

def count_percet_solution(response_data, lowest_energy):
    #print (lowest_energy, end=" ")
    num_of_correct_solution = 0
    for data in response_data:
        #print(data, end=" ")
        if (data.energy == lowest_energy):
            num_of_correct_solution += data.num_occurrences
    #print("Nums of correct solution: ", num_of_correct_solution, end="\n")
    #print("Phan tram giai duoc: ", num_of_correct_solution / 1000)
    return num_of_correct_solution

    
if __name__ == "__main__":
    # Create a simple graph
    # G1 = nx.Graph()
    # G1.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (4, 5)])
    input_folder = "input"  # Thư mục chứa các file TXT
    file_to_read = "graph1.txt"  # File cần đọc

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
        #print(f"Edges of the graph in {file_to_read}: {G.edges()}")
    
    # Generate QUBO
    Q = create_mis_qubo(G, penalty_weight=2.0)
    #print(Q)

    chainstrength = 8
    numruns = 1000
    # start_time = time.time()
    sampler = EmbeddingComposite(DWaveSampler(token='DEV-b28f5c26b9419829978caa8899867ab5c25f9802'))
    response = sampler.sample(Q,
                               chain_strength=chainstrength,
                               num_reads=numruns,
                               annealing_time=1,
                               label='Maximum Independent Set')

  
    ##sampleset = dimod.SimulatedAnnealingSampler()
    # Measure time for Simulated Annealing
    
    #sampler = SimulatedAnnealingSampler()

    response = sampler.sample(Q, num_reads = 1000)
    
    #dwave.inspector.show(response)

    lowest_energy = response.first.energy
    
    for data in response.data():
        print(data)
    print("/n", "---------------------------------")
    print("Nang luong thap nhat la: ", lowest_energy)
    print("So lan vi pham rang buoc: ", count_num_penalty(response.data(), G))
    print("So solution dung la:", count_percet_solution(response.data(), lowest_energy))
    print("Phan tram so cau tra loi dung: ", count_percet_solution(response.data(), lowest_energy)/1000)
    print("Best solutions are {}% of samples.".format(len(response.lowest(atol=0.5).record.energy)/10))
    print(response.info["timing"])

    #solution = response.first    

    # Đường dẫn đến thư mục output và file output_1.json
    
    output_folder = "output"
    file_to_write = "output_1.json"
    file_path_write = os.path.join(output_folder, file_to_write)

    # Tạo thư mục output nếu chưa có
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Chuẩn bị dữ liệu để ghi vào file JSON
    output_data = []

    # Chuyển đổi dữ liệu từ Sample thành dạng có thể lưu vào JSON
    for data in response.data():
        #sample = data.sample
        energy = data.energy
        num_occurrences = data.num_occurrences
        chain_break_fraction = data.chain_break_fraction
        sample_info = {
        #"sample": data.sample,  # Đưa dictionary sample vào (các giá trị như {0: 0, 1: 1, ...})
        "energy": energy,
        "num_occurrences": int(num_occurrences),
        "chain_break_fraction": chain_break_fraction
        }
        output_data.append(sample_info)
    
    result_info = {
        "Nang luong thap nhat": lowest_energy,
        "So lan vi pham rang buoc": count_num_penalty(response.data(), G),
        "So solution dung la": int(count_percet_solution(response.data(), lowest_energy)),
        "Phan tram so cau tra loi dung": count_percet_solution(response.data(), lowest_energy)/1000,
        "Best solutions of samples %": format(len(response.lowest(atol=0.5).record.energy)/10)
    }
    output_data.append(result_info)

    # Lưu dữ liệu vào file JSON
    try:
        with open(file_path_write, "w") as output_file:
            json.dump(output_data, output_file, indent=4)
        print(f"Results have been saved to {file_path_write}")
    except Exception as e:
        print(f"Error while writing JSON: {e}")
