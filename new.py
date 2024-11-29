from ortools.linear_solver import pywraplp


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
        return independent_set, total_weight
    else:
        return None, 0


if __name__ == "__main__":
    ans = maximum_weighted_independent_set([1, 1, 1], [(0, 2), (1, 2)])
    print(ans)