from flask import Flask, render_template, request, jsonify
import sys, webbrowser, time, random, threading
import numpy as np
from scipy import linalg
from pulp import *
import time
import logging
from mpi4py import MPI

#############################################
# Logging Configurations
############################################
np.set_printoptions(threshold=np.nan, linewidth=1000)

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S',
                    level=logging.DEBUG)
minesweeper_logger = logging.getLogger("minesweeper_logger")
logging.getLogger("pulp").setLevel(logging.WARNING)

#########################################################
# MPI Configurations
#########################################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#################################################
# Parallelization Methods
##################################################
def choose_rows(U, b, num_threads):
    """ 
    Chooses the first set of variables to find feasible solutions for. 
    U: numpy matrix from LU decomposition of the binary equation matrix.
    num_threads: number of threads for which we select the matrix.
    pos_var: Maps variables to true position on board.
    """
    possible_rows = []
    delete_rows = []
    selected_b = []

    for i in range(len(U)):
        row = U[i]
        unique, counts = np.unique(row, return_counts=True)
        counts_map = dict(zip(unique, counts))

        if 1.0 not in counts_map:
            counts_map[1.0] = 0
        if -1.0 not in counts_map:
            counts_map[-1.0] = 0
        if 0.0 not in counts_map:
            counts_map[0.0] = 0

        if counts_map[1.0] + counts_map[-1.0] == 1:
            continue  # We don't want to pick this row
        elif counts_map[0.0] == len(U[0]):
            delete_rows.append(i)
        else:
            possible_rows.append(row)
            selected_b.append(b[i])
    np.delete(U, delete_rows, axis=0)
    minesweeper_logger.debug("Possible rows \n%s", possible_rows)
    return np.array(possible_rows[:num_threads]), selected_b[:num_threads]  # Return all the rows.

#################################################
# Serial code optimization
################################################
def custom_reduction(u):
    """
    Custom reduction function for the LU decomposed matrix 'u'. 
    The goal of the reduction is to first find rows that contain a single 1. 
    For such rows, look at all rows above that contain varaibles not uniquely solved,
    and subtract row_above - current_row. This simplifies the binary programming 
    and improves its runtime.
    :param u: the U matrix from the LU decomposition of the augmented binary 
    equation matrix.
    :return: The U matrix after performing this custom_reduction to the matrix.
    """
    solved_rows = []  # Rows that are uniquely solved.

    for i in range(len(u)):  # Iterate through the rows of u
        row = u[i]  # Get the row
        unique, counts = np.unique(row, return_counts=True)
        counts_map = dict(zip(unique, counts))

        if 1.0 in counts_map:
            if counts_map[1.0] == 2 and row[-1] == 1:  # If there is a single 1 and a 1 at the end.
                # Then reduce with row
                solved_rows.append(i)  # This row is fully solved.
                u = reduce_row(u, row, i, solved_rows)
            elif counts_map[1.0] == 1 and row[-1] == 0:  # If there is a single 1 and a 0 at the end.
                solved_rows.append(i)  # This row is fully solved
                u = reduce_row(u, row, i, solved_rows)

        if -1.0 in counts_map:
            if counts_map[-1.0] == 1 and row[-1] == 1:
                solved_rows.append(i)
                u = reduce_row(u, row, i, solved_rows, minus_one=True)
            elif counts_map[-1.0] == 1 and row[-1] == 0:
                solved_rows.append(i)
                u = reduce_row(u, row, i, solved_rows, minus_one=True)
        else:
            pass

    return u


def reduce_row(u, row, row_num, solved_rows, minus_one=False):
    """
    Helper method for custom_reduction. This is called on each row that should
    be reducing the rows above it.
    :param u: The U matrix is a partially reduced matrix. 
    :param row: The row for which this method was called.
    :param row_num: The index of the row.
    :param solved_rows: A list of solved rows
    :param minus_one: A flag that specifies whether or not the reduction is for
    'row' that contains '1' or '-1'
    :return: Further reduced 'u' matrix.
    """
    search_for = 1

    if minus_one:
        search_for = -1

    index = None  # Find the index of the '1'
    for i, elem in enumerate(row):
        if elem == search_for:
            index = i
            break
    else:
        index = None

    for i in range(row_num):
        other_row = u[i]

        if other_row[index] == search_for and i not in solved_rows:
            u[i] = u[i] - row
    return u


def delete_zero_cols(chosen_rows, pos_var):
    """
    Given the chosen rows to solve partially in the serial portion of the code,
    Removes the columns that contain pure zeros. Correspondingly, updates pos_var.
    :param chosen_rows: 
    :param pos_var: A customized mapping of variable numbers (x_1, x_2, ...) used 
    in binary programming to their actual tile index on the board.
    :return: Modifies and returns chosen_rows and pos_var.
    """
    delete_columns = []

    for i in range(len(chosen_rows.T)):
        col = chosen_rows.T[i]

        for elem in col:
            if elem != 0:
                break
        else:
            delete_columns.append(i)

    if len(delete_columns) > 0:
        chosen_rows = np.delete(chosen_rows, delete_columns, 1)

    new_pos_var = []
    for i in range(len(pos_var)):
        if i not in delete_columns:
            new_pos_var.append(pos_var[i])

    return chosen_rows, new_pos_var


#####################################################
# Helper Methods
#####################################################
def is_unopened(board, index):
    """Takes in a tuple 'index', and board.
    Returns true if tile is unopened"""
    return board[index[0]][index[1]] == -1


def is_opened(board, index):
    return not is_unopened(board, index)

def set_array_for_scatter(arr):
    num_processors = comm.size
    new_arr = [[] for i in range(num_processors)] # This MUST have size comm.size by the end.

    for i, elem in enumerate(arr):
        array_index = i % num_processors
        new_arr[array_index].append(elem)

    return new_arr

####################################################
# Central method. Selects which tile to open
#################################################
def solve(board):
    linear_mat_reduced = None
    edge_num_reduced = None
    new_pos_var = None
    pos_var = None

    minesweeper_logger.debug("I am rank {}".format(rank))
    comm.Barrier()
    minesweeper_logger.debug("I am rank {}".format(rank))

    # if rank == 0:
    if is_unopened(board, (0, 0)):
        return [0, 0]
    # else:
    #     if is_unopened(board, (0, 0)):
    #         return [0, 0]

    # Prepare the board by getting the linear equations and a mapping of variable to tiles.
    if rank == 0:
        linear_mat, edge_num, pos_var = prepare(board)
        linear_mat_np = np.matrix(linear_mat)  # Convert to np matrix.
        edge_num_np = np.matrix(edge_num)  # Convert to np matrix.

    if rank == 0:
        minesweeper_logger.debug("Edge num is: \n%s", edge_num_np)
        # Augment the matrix to do LU decomposition.
        linear_matrix_augmented = np.hstack((linear_mat_np, np.array(edge_num_np).T))
        minesweeper_logger.debug("Linear equations augmented matrix is: \n%s", linear_matrix_augmented)
        pl, u = linalg.lu(linear_matrix_augmented, permute_l=True)  # Perform LU decomposition. U is gaussian eliminated.
        minesweeper_logger.debug("U matrix of linear equations joined matrix is: \n%s", u)

        start_custom_reduction = time.time()
        reduced_u = custom_reduction(u)
        end_custom_reduction = time.time()
        minesweeper_logger.debug("Reduced U matrix of lin. eqns joined matrix is: \n%s", reduced_u)
        minesweeper_logger.info("Custom reduction took \n%s", end_custom_reduction - start_custom_reduction)

        edge_num_reduced = list(reduced_u[:, -1])
        linear_mat_reduced = reduced_u[:, :-1]

    if rank == 0:
        # Select rows that we want to solve as a subproblem in the serial part.
        selected_rows, selected_b = choose_rows(linear_mat_reduced, edge_num_reduced, num_threads=2)
        selected_rows, new_pos_var = delete_zero_cols(selected_rows, pos_var)

        minesweeper_logger.debug("Selected rows \n%s", selected_rows)
        minesweeper_logger.debug("New b \n%s", selected_b)
        minesweeper_logger.debug("New pos var \n%s", new_pos_var)

    partial_feasible_solution = []
    # First bin_programming solve subproblem
    if rank == 0 and len(selected_b) != 0:
        minesweeper_logger.debug("selected rows \n%s", selected_rows)
        minesweeper_logger.debug("selected b \n%s", selected_b)
        start_partial_bp_solver = time.time()
        partial_feasible_solution = solve_binary_program(selected_rows, selected_b)
        end_partial_bp_solver = time.time()
        minesweeper_logger.info("Partial feasible sol is \n%s", partial_feasible_solution)
        minesweeper_logger.info("Partial BP solver took \n%s", end_partial_bp_solver - start_partial_bp_solver)
        minesweeper_logger.info("Partial feas sol len \n%s", len(partial_feasible_solution))


    # partial_feasible_solution. Currently, it's NONE in all ranks except for 0.
    # inside of rank 0 --- 1. its length is more than 2, or its less than 2.
    # IF the length is < 2... i just wanna do serial.
    # If the length > 2 --- then I wanna scatter.
    # minesweeper_logger.debug("I am rank {}".format(rank))
    minesweeper_logger.debug("I am rank {}".format(rank))
    comm.Barrier()
    minesweeper_logger.debug("I am rank {}".format(rank))
    # if rank == 0:
    minesweeper_logger.debug("Partial feasible solution length is, %s", len(partial_feasible_solution))
    # if len(partial_feasible_solution) > 2:
    to_scatter = set_array_for_scatter(partial_feasible_solution)
    minesweeper_logger.debug("Going to scatter from rank {}".format(rank))
    partial_feasible_solution = comm.scatter(to_scatter, root=0)
    # Broadcast all the data required.
    minesweeper_logger.info("Scattered partial_feasible_solution, now broadcasting from rank 0...")
    comm.Barrier()
    minesweeper_logger.info("Finished with barrier?")
    linear_mat_reduced = comm.bcast(linear_mat_reduced, root=0)
    edge_num_reduced = comm.bcast(edge_num_reduced, root=0)
    new_pos_var = comm.bcast(new_pos_var, root=0)
    pos_var = comm.bcast(pos_var, root=0)
    if rank == 0:
        minesweeper_logger.info("Finished broadcast from rank 0...")

    minesweeper_logger.debug("I am rank {}".format(rank))
    comm.Barrier()
    minesweeper_logger.debug("I am rank {}".format(rank))


    parallel_feasible_solutions = [] # All procs initailize this to be entry.
    print("I am rank {}".format(rank))
    if partial_feasible_solution != None and len(partial_feasible_solution) > 0: # This means that the root sent me something
        for j, sol in enumerate(partial_feasible_solution):
            # All processors go through their list of partial_feasible_solutions
            # set timer
            time_parallel_proc = time.time() # All processors time.
            constraints = [(pos_var.index(new_pos_var[i]), sol[i]) for i in range(len(sol))]
            parallel_feasible_solutions.extend(solve_binary_program(linear_mat_reduced, edge_num_reduced, constraints)) # All processors solve.
            end_time_parallel_proc = time.time()

            print("Proc {} took \n%s".format(rank), end_time_parallel_proc - time_parallel_proc) # All processors log.


    minesweeper_logger.debug("I am rank {}".format(rank))
    comm.Barrier()
    minesweeper_logger.debug("I am rank {}".format(rank))

    parallel_feasible_solutions = comm.allgather(parallel_feasible_solutions)
    # if rank == 0:
    minesweeper_logger.info("Finished gathering at rank {}...".format(rank))

    # if rank == 0:
    if len(partial_feasible_solution) <= 2:
        start_bp_solver = time.time()
        serial_feasible_soln = solve_binary_program(linear_mat_reduced, edge_num_reduced)
        end_bp_solver = time.time()
        minesweeper_logger.info("Had to use serial solver. Took time %s".format(end_bp_solver - start_bp_solver))
        minesweeper_logger.debug("Length of serial feasible solution: \n%s", serial_feasible_soln)
        final_feasible_solution = serial_feasible_soln
    else:
        final_feasible_solution = parallel_feasible_solutions
        minesweeper_logger.debug("Length of parallel feasible solution: \n%s", len(parallel_feasible_solutions))

    probabilities = np.sum(final_feasible_solution, axis=0)
    # TODO: Make a 2d matrix out of probabilities so that we can display it on the grid.
    tile_to_open = pos_var[np.argmin(probabilities)]
    length_of_row = len(board[0])
    y_index = tile_to_open / length_of_row
    x_index = tile_to_open % length_of_row

    return [x_index, y_index]
    # else:
        # return [0, 0]

#####################################################
# Creates equations
#######################################################
def prepare(board):
    """
    :param board: 
    :return: 
        :linear_mat: Linear equation matrix, to be further processed and solved using
        the *binary* equation solver. This matrix is NOT augmented, and only contains the equations.
        :edge_num: This is the value vector (array) corresponding to linear_equation matrix. Used in
        creating the augmented matrix. This value will always be the value of the opened tiles. 
        :pos_var: A customized mapping from variable numbers (i.e. columns of linear_mat) to the actual
        tile indices on the board. For instance, variable x_3 might correspond to the 26th tile on the board.
    """
    num_cols = len(board[0])
    num_rows = len(board)
    pos_var = []
    edge_num = []
    idx_mat = []
    for i, j in itertools.product(range(num_cols), range(num_rows)):
        if board[j][i] > 0:
            var_idx = []
            for offset_x, offset_y in itertools.product([-1, 0, 1], [-1, 0, 1]):
                neighbor_x = i + offset_x
                neighbor_y = j + offset_y
                if neighbor_x >= 0 and neighbor_y >= 0 and neighbor_x < num_cols and neighbor_y < num_rows and \
                                board[neighbor_y][neighbor_x] == -1:
                    cur_neighbor_idx = neighbor_y * num_cols + neighbor_x
                    var_idx.append(cur_neighbor_idx)
                    if cur_neighbor_idx not in pos_var:
                        pos_var.append(cur_neighbor_idx)
            if len(var_idx) > 0:
                cur_number_idx = j * num_cols + i
                edge_num.append(board[cur_number_idx / num_cols][cur_number_idx % num_cols])
                idx_mat.append([pos_var.index(idx) for idx in var_idx])

    linear_mat = np.zeros((len(edge_num), len(pos_var)), dtype=np.int8)
    for i in range(len(edge_num)):
        for j in idx_mat[i]:
            linear_mat[i][j] = 1
    edge_num = np.array(edge_num)
    return linear_mat, edge_num, pos_var

#######################################################
# Binary program solver
#######################################################
def solve_binary_program(linear_mat, edge_num, constraints=[]):
    """
    Solves the given binary program, and returns ALL feasible solutions. 
    The PULP solver used as a subroutine does not actually have a facility to return
    the set of all feasible solutions. This method does so using a while(True) loop.
    One the PULP solver returns a set of solutions, the binary program is restricted
    to add another constraint that prevents those variables from taking those exact values. 
    For instance, suppose that the pulp solver returns the values
    S_1 = (x_1 = 1, x_2 = 1, x_3 = 0, x_4 = 1). A new constraint is added that restricts
    x_1 + x_2 + x_4 from being 3 again. The nature of minesweeper is such that
    S_2 = (x_1 = 1, x_2 = 1, x_3 = 1, x_4 = 1) will not be a feasible solution if S_1 is
     a feasible solution. 
    :param linear_mat: Binary equation matrix. Non-augmented. In an equation Ax = b, the A matrix.
    :param edge_num: Value vector. In Ax = b, the b vector. 
    :param constraints: Optional, additional constraints imposed on the solver. 
    These additional constraints are the heart of the parallelization. This binary_program is
     first used to find a subset of the feasible solutions. Then, suppose that the variables we
     gather solutions for are (x_1, x_2, x_3), and we get 6 possible solutions. These are scattered to 
     6 different processors, and each one solves binary_programs with additional constraints constraining
     x_1, x_2, x_3 to be the values that are assigned to said processor. 

    :return: List of feasible solutions for the variables. The mapping from the variable solution to
    the corresponding tile on the board is maintained by pos_var.
    """
    prob = LpProblem("oneStep", LpMinimize)
    var = []
    for i in range(linear_mat.shape[1]):
        var.append(LpVariable('a' + str(i), lowBound=0, upBound=1, cat='Integer'))

    if len(constraints) > 0:
        for i in range(len(constraints)):
            var_num, val = constraints[i]
            var[var_num] = val

    constraints_var = []
    for j in range(len(edge_num)):
        constraint = LpVariable('b' + str(j), lowBound=0, upBound=0, cat='Integer')
        for k in range(linear_mat.shape[1]):
            constraint += var[k] * linear_mat[j][k]
        constraints_var.append(constraint)

    for j in range(len(edge_num)):
        prob += constraints_var[j] == edge_num[j], "constraint " + str(j + 1)

    feas_sol = []
    while True:
        prob.solve()
        if LpStatus[prob.status] == "Optimal":
            sol = []
            for i in range(len(var)):
                sol.append(value(var[i]))
            feas_sol.append(sol)
            prob += lpSum([var[i] for i in range(len(var)) if value(var[i]) == 1]) <= len(
                [var[i] for i in range(len(var)) if value(var[i]) == 1]) - 1
        else:
            break
    feas_sol = np.array(feas_sol)
    return feas_sol


##############################################################
## Web Framework
##############################################################

app = Flask(__name__, static_folder='static', static_url_path='')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/solve_next', methods=['POST'])
def solve_next():
    data = request.get_json()
    # print(data)
    return jsonify(solve(data))


if __name__ == '__main__':
    # if rank == 0:
        port = 5000 + random.randint(0, 999)
        url = "http://127.0.0.1:{0}".format(port)
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
        app.run(port=port, debug=False)
