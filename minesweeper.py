from flask import Flask, render_template, request, jsonify
import sys, webbrowser, time, random, threading, argparse
import numpy as np
from scipy import linalg
from pulp import *
import time
import logging

import thread

import time

np.set_printoptions(threshold=np.nan, linewidth=1000)

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S',
                    level=logging.INFO)
minesweeper_logger = logging.getLogger("minesweeper_logger")
logging.getLogger("pulp").setLevel(logging.WARNING)


NUM_THREADS = 2


clear_grid = []

class myThread (threading.Thread):
   def __init__(self, threadID, name, counter, linear_mat_reduced, edge_num_reduced, partial_feasible_sol, pos_var, new_pos_var):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
      self.linear_mat_reduced = linear_mat_reduced
      self.edge_num_reduced = edge_num_reduced
      self.partial_feasible_sol = partial_feasible_sol
      self.pos_var = pos_var
      self.new_pos_var = new_pos_var

   def run(self):
      # Get lock to synchronize threads
      threadLock = threading.Lock()
      threadLock.acquire()
      time_parallel_proc = time.time()
      self.feas_sol = parallel_solving(self.linear_mat_reduced, self.edge_num_reduced, self.partial_feasible_sol, self.pos_var, self.new_pos_var, self.threadID)
      end_time_parallel_proc = time.time()
      minesweeper_logger.info("Proc {} took \n%s".format(self.threadID), end_time_parallel_proc - time_parallel_proc)
      # Free lock to release next thread
      threadLock.release()

   def get_value(self):
      return self.feas_sol

def parallel_solving(linear_mat_reduced, edge_num_reduced, partial_feasible_sol, pos_var, new_pos_var, threadID):
    feas_sol = []
    for j, sol in enumerate(partial_feasible_sol):
        if j % NUM_THREADS == threadID:
            constraints = [(pos_var.index(new_pos_var[i]), sol[i]) for i in range(len(sol))]
            feas_sol.extend(solve_binary_program(linear_mat_reduced, edge_num_reduced, constraints))
    return feas_sol

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
    if len(possible_rows) >= num_threads:
        return np.array(possible_rows[-1-num_threads:-1]), selected_b[-1-num_threads:-1]  # Return all the rows.
    else:
        return np.array(possible_rows[:]), selected_b[:]

def choose_clear_grids(u):
    clear_grids_index = []
    solved_rows = []  # Rows that are uniquely solved.

    for i in range(len(u)):  # Iterate through the rows of u
        row = u[i]  # Get the row
        unique, counts = np.unique(row, return_counts=True)
        counts_map = dict(zip(unique, counts))

        if 1.0 not in counts_map:
            counts_map[1.0] = 0
        if -1.0 not in counts_map:
            counts_map[-1.0] = 0
        if 0.0 not in counts_map:
            counts_map[0.0] = 0

        if counts_map[1.0] > 0:
            if counts_map[1.0] == 2 and row[-1] == 1:  # If there is a single 1 and a 1 at the end.
                # Then reduce with row
                solved_rows.append(i)  # This row is fully solved.
                u = reduce_row(u, row, i, solved_rows)
            elif counts_map[1.0] + counts_map[-1.0] == 1 and row[-1] == 0:  # If there is a single 1 and a 0 at the end.
                solved_rows.append(i)  # This row is fully solved
                clear_grids_index.append(i)
                u = reduce_row(u, row, i, solved_rows)

        if counts_map[-1.0] > 0:
            if counts_map[-1.0] == 1 and row[-1] == 1:
                solved_rows.append(i)
                u = reduce_row(u, row, i, solved_rows, minus_one=True)
            elif counts_map[1.0] + counts_map[-1.0] == 1 and row[-1] == 0:
                solved_rows.append(i)
                clear_grids_index.append(i)
                u = reduce_row(u, row, i, solved_rows, minus_one=True)
        else:
            pass
    return clear_grids_index

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


def solve_step(board):
    if is_unopened(board, (0, 0)):
        return {'grids':[[0, 0]], 'times':[0, 0, 0]}

    # Prepare the board by getting the linear equations and a mapping of variable to tiles.
    linear_mat, edge_num, pos_var = prepare(board)
    linear_mat_np = np.matrix(linear_mat)  # Convert to np matrix.
    edge_num_np = np.matrix(edge_num)  # Convert to np matrix.

    minesweeper_logger.debug("Edge num is: \n%s", edge_num_np)
    # Augment the matrix to do LU decomposition.
    linear_matrix_augmented = np.hstack((linear_mat_np, np.array(edge_num_np).T))

    minesweeper_logger.debug("Linear equations augmented matrix is: \n%s", linear_matrix_augmented)
    pl, u = linalg.lu(linear_matrix_augmented, permute_l=True)  # Perform LU decomposition. U is gaussian eliminated.

    minesweeper_logger.debug("U matrix of linear equations joined matrix is: \n%s", u)

    clear_grid_index = choose_clear_grids(u)
    if len(clear_grid_index) > 0:
        print "I am sure"
        clear_grid_early = []
        for i in range(len(clear_grid_index)):
            tile_to_open = pos_var[clear_grid_index[i]]
            length_of_row = len(board[0])
            y_index = tile_to_open / length_of_row
            x_index = tile_to_open % length_of_row
            clear_grid_early.append([x_index, y_index])
        # return clear_grid_early
        return {'grids':clear_grid_early, 'times':[0, 0, 0]}
    else:
        print "I am guessing"

    time_custom_reduction = 0
    time_partial_bp_solver = 0
    time_bp_solver = 0

    start_custom_reduction = time.time()
    reduced_u = custom_reduction(u)
    time_custom_reduction = time.time() - start_custom_reduction
    minesweeper_logger.debug("Reduced U matrix of lin. eqns joined matrix is: \n%s", reduced_u)
    minesweeper_logger.info("Custom reduction took \n%s", time_custom_reduction)

    edge_num_reduced = list(reduced_u[:, -1])
    linear_mat_reduced = reduced_u[:, :-1]


    # Select rows that we want to solve as a subproblem in the serial part.
    selected_rows, selected_b = choose_rows(linear_mat_reduced, edge_num_reduced, num_threads = NUM_THREADS/4)
    selected_rows, new_pos_var = delete_zero_cols(selected_rows, pos_var)

    minesweeper_logger.debug("Selected rows \n%s", selected_rows)
    minesweeper_logger.debug("New b \n%s", selected_b)
    minesweeper_logger.debug("New pos var \n%s", new_pos_var)

    # First bin_programming solve subproblem
    if len(selected_b) != 0:
        minesweeper_logger.debug("selected rows \n%s", selected_rows)
        minesweeper_logger.debug("selected b \n%s", selected_b)
        start_partial_bp_solver = time.time()
        partial_feasible_sol = solve_binary_program(selected_rows, selected_b)
        time_partial_bp_solver = time.time() - start_partial_bp_solver
        minesweeper_logger.info("Partial feasible sol is \n%s", partial_feasible_sol)
        minesweeper_logger.info("Partial BP solver took \n%s", time_partial_bp_solver)

        # Imagine that we have distributed
        feas_sol = []
        minesweeper_logger.info("Partial feas sol len \n%s", len(partial_feasible_sol))
    else:
        partial_feasible_sol = []

    if len(partial_feasible_sol) > 1:

        threads = []
        # Create new threads
        for i in range(NUM_THREADS):
            threads.append(myThread(i, "Thread"+str(i), i, linear_mat_reduced, edge_num_reduced, partial_feasible_sol, pos_var, new_pos_var))

        # Start new Threads
        for i in range(NUM_THREADS):
            threads[i].start()

        # Wait for all threads to complete
        for t in threads:
            t.join()
        for i in range(NUM_THREADS):
            feas_sol += threads[i].get_value()

    if len(partial_feasible_sol) <= 1:
        start_bp_solver = time.time()
        serial_feasible_soln = solve_binary_program(linear_mat_reduced, edge_num_reduced)
        time_bp_solver = time.time() - start_bp_solver
        feas_sol = serial_feasible_soln
        minesweeper_logger.info("BP Solver took \n%s", time_bp_solver)

    minesweeper_logger.debug("Length of feasible solution from reduction: \n%s", len(feas_sol))
    #minesweeper_logger.debug("Length of feasible solution from serial: \n%s", serial_feasible_soln)

    probabilities = np.sum(feas_sol, axis=0)

    grids = []
    for ind, prob in enumerate(list(probabilities)):
        if prob == 0:
            tile_to_open = pos_var[ind]
            length_of_row = len(board[0])
            y_index = tile_to_open / length_of_row
            x_index = tile_to_open % length_of_row
            grids.append([x_index, y_index])
    if len(grids) == 0:
        tile_to_open = pos_var[np.argmin(probabilities)]
        length_of_row = len(board[0])
        y_index = tile_to_open / length_of_row
        x_index = tile_to_open % length_of_row
        grids.append([x_index, y_index])

    return {'grids':grids, 'times':[time_bp_solver, time_partial_bp_solver, time_custom_reduction]}


def solve(board):
    global clear_grid
    times = [0, 0, 0]
    if len(clear_grid) == 0:
        output = solve_step(board)
        print(output)
        times = output['times']
        clear_grid = output['grids']
    next_move = clear_grid[-1]
    print(next_move)
    del clear_grid[-1]
    return [next_move[0], next_move[1]] + times


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
    options = {'autostart': app.config.get('autostart')}
    return render_template('index.html', options=options)


@app.route('/api/solve_next', methods=['POST'])
def solve_next():
    data = request.get_json()
    solution = solve(data["board"])
    solution.append(data["gameId"])
    return jsonify(solution)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--autostart", help="Start auto-solve on launch", action="store_true")
    parser.add_argument("--deploy", help="Host over network", action="store_true")
    parser.add_argument("-p", dest="p", default=10, type=int, help="number of threads")
    args = parser.parse_args()
    global NUM_THREADS
    NUM_THREADS = args.p
    app.config['autostart'] = args.autostart

    if args.deploy:
        app.run(host= '0.0.0.0')
    else:
        port = 5000 + random.randint(0, 999)
        url = "http://127.0.0.1:{0}".format(port)
        threading.Timer(0.5, lambda: webbrowser.open(url) ).start()
        app.run(port=port, debug=False)
