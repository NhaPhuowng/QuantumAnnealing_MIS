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
        res = data.sample
        test_list = []  # khoi tao danh sach cac dinh duoc chon
        penalty = 0  # khoi tao so lan vi pham rang buoc

        # tao list cac dinh duoc chon
        for i in res.keys():
            #print(res.get(i), end=" ")
            if res.get(i) == 1:
                 test_list.append(i)

        # tu list do tao nen cac cap canh kha thi, dem so cap thuoc do thi G
        possible_edge = list(combinations(test_list, 2))
        for x in possible_edge:
            if (x in G.edges):
                penalty += 1
                total_penalty += 1
    return total_penalty

def count_percet_solution(response_data, lowest_energy):
    num_of_correct_solution = 0
    for data in response_data:
        if (data.energy == lowest_energy):
            num_of_correct_solution += data.num_occurrences
    return num_of_correct_solution

def count_denity_graph(n, num_edges):
    return 2 * num_edges / (n * (n - 1))

def create_random_graph(n, num_edges):
    # Tạo danh sách các cạnh trong đồ thị đầy đủ
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))

    # Chọn ngẫu nhiên 147 cạnh từ danh sách
    random.shuffle(edges)
    selected_edges = edges[:num_edges]
    
    input_folder = "input"  # Thư mục chứa các file TXT
    file_to_read = "graph_10.txt"  # File cần đọc

    # Đường dẫn đầy đủ đến file
    file_path = os.path.join(input_folder, file_to_read)
    # In các cạnh theo yêu cầu
    if os.path.exists(file_path):
        with open(file_path, "w") as f:
            for edge in selected_edges:
                f.write(f"{edge[0]} {edge[1]}\n")

    print(f"Đồ thị với {n} đỉnh và {num_edges} cạnh đã được lưu vào file {file_path}")

    
def maximum_weighted_independent_set(weights, edges):
    # Create the solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return None

    num_nodes = len(weights)

    # Decision variables: x[i] = 1 if node i is in the independent set, 0 otherwise
    x = {}
    for i in range(num_nodes):
        x[i] = solver.BoolVar(f'x[{i}]')

    # Objective function: maximize sum of weights * x[i]
    objective = solver.Objective()
    for i in range(num_nodes):
        objective.SetCoefficient(x[i], weights[i])
    objective.SetMaximization()

    # Constraints: For each edge (i, j), ensure that x[i] + x[j] <= 1
    for (i, j) in edges:
        solver.Add(x[i] + x[j] <= 1)

    # Solve the problem
    status = solver.Solve()
    print('TimeRunning %f' % (solver.wall_time() / 1000.0))
    print('Problem solved in %d iterations' % solver.iterations())
    # Check if a solution was found
    if status == pywraplp.Solver.OPTIMAL:
        print('OptimalSolution', solver.Objective().Value())
        independent_set = [i for i in range(num_nodes) if x[i].solution_value() == 1]
        total_weight = sum(weights[i] for i in independent_set)
        return total_weight
    else:
        return 0

if __name__ == "__main__":
    
    # create radom graph and save to folder input
    #create_random_graph(23, 51)
    
    # # create graph
    input_folder = "input"  # Thư mục chứa các file TXT
    file_to_read = "graph_31.txt"  # File cần đọc

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
                
    # visualize the graph and save to folder image
    # output_folder = "image"
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    
    # plt.figure(figsize=(8, 6))
    # nx.draw(G, with_labels=True, node_color='skyblue', node_size=3000, font_size=10, font_weight='bold', edge_color='gray')

    # output_file_path = os.path.join(output_folder, "map_10.png")
    # plt.savefig(output_file_path)
    # print(f"Đã lưu đồ thị vào {output_file_path}")
    
    # using ortool
    list_of_ones = [1] * len(G.nodes)
    res_ortools = maximum_weighted_independent_set(list_of_ones, G.edges())
    #print("Ortool: ", res_ortools)
    
    # Generate QUBO
    penalty_weigth_num = 5.0
    Q = create_mis_qubo(G, penalty_weight=penalty_weigth_num)
    print(Q)
    
    # Quantum Annealing
    chainstrength = 8
    numruns = 1000
    annealingTime = 100
    sampler = EmbeddingComposite(DWaveSampler(token='DEV-b28f5c26b9419829978caa8899867ab5c25f9802'))
    response = sampler.sample(Q,
                               chain_strength=chainstrength,
                               num_reads=numruns,
                               annealing_time=annealingTime,
                               label='Maximum Independent Set')
    
    # Simulated Annealing
    ##sampleset = dimod.SimulatedAnnealingSampler()
    # Measure time for Simulated Annealing
    #sampler = SimulatedAnnealingSampler()

    response = sampler.sample(Q, num_reads = 1000)
    
    #dwave.inspector.show(response)
    
    # using Exact Solver
    # solver = dimod.ExactSolver()
    # sampleset = solver.sample(Q)
    # min_energy_ExactSolver = sampleset.first.energy
    

    lowest_energy = response.first.energy
    lowest_energy_orTools = - res_ortools
    
    # print data to terminal
    for data in response.data():
        print(data)
    print("-------------------------------------------")
    print("Gamma:", penalty_weigth_num)
    print("Annealing_time:", annealingTime)
    print("So dinh:", G.number_of_nodes())
    print("So canh:", G.number_of_edges())
    print("Mat do do thi:", count_denity_graph(G.number_of_nodes(), G.number_of_edges()))
    print("Nang luong thap nhat bi sai ban dau la: ", lowest_energy)
    print("So lan vi pham rang buoc: ", count_num_penalty(response.data(), G))
    print("So solution dung la:", count_percet_solution(response.data(), lowest_energy_orTools))
    print("Phan tram so cau tra loi dung: ", count_percet_solution(response.data(), lowest_energy_orTools)/10)
    print("Best solutions are {}% of samples.".format(len(response.lowest(atol=0.5).record.energy)/10))
    print(response.info["timing"])
    #print("Nang luong thap nhat theo Exact Solver: ", min_energy_ExactSolver)
    print("Nang luong thap nhat va loi giai toi uu ortools: ", res_ortools)

    #solution = response.first    

    # Save data to folder out_put
    output_folder = "output_gamma5_100"
    file_to_write = "output_31_gamma5.json"
    file_path_write = os.path.join(output_folder, file_to_write)
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
        "Gamma": penalty_weigth_num,
        "Annealing_time": annealingTime,
        "So dinh": G.number_of_nodes(),
        "So canh": G.number_of_edges(),
        "Mat do do thi": count_denity_graph(G.number_of_nodes(), G.number_of_edges()),
        "Nang luong thap nhat bi sai ban dau": lowest_energy,
        "So lan vi pham rang buoc": count_num_penalty(response.data(), G),
        "So solution dung la": int(count_percet_solution(response.data(), lowest_energy_orTools)),
        "Phan tram so cau tra loi dung": count_percet_solution(response.data(), lowest_energy_orTools)/1000,
        "Best solutions of samples %": format(len(response.lowest(atol=0.5).record.energy)/10),
        "Thoi gian chay": response.info["timing"],
        #"Nang luong thap nhat theo ExactSolver": min_energy_ExactSolver,
        "Nang luong thap nhat va cac loi giai toi uu theo ortools": res_ortools
    }
    output_data.append(result_info)

    # Lưu dữ liệu vào file JSON
    try:
        with open(file_path_write, "w") as output_file:
            json.dump(output_data, output_file, indent=4)
        print(f"Results have been saved to {file_path_write}")
    except Exception as e:
        print(f"Error while writing JSON: {e}")
        
    # print(f"Number of logical variables: {len(embedding.keys())}")
    # print(f"Number of physical qubits used in embedding: {sum(len(chain) for chain in embedding.values())}")
