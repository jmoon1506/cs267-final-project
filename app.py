#!/usr/bin/env python

from flask import Flask, render_template, request, jsonify
import sys, webbrowser, time, random, threading, argparse
import numpy as np
from scipy import linalg
from pulp import *
import time
import logging
import msboard
import csv

import thread
import multiprocessing
import os
import time
import math

from mpi4py import MPI

np.set_printoptions(threshold=np.nan, linewidth=1000)

# threadLock = multiprocessing.Lock()

NUM_THREADS = 2

# GLOBALS
clear_grid = []
gameId = 0
clear_grid_distributed = []

board = msboard.MSBoard(16, 32, 99)
minesweeper_logger = None
args = None

def log_debug(msg, val=None):
    if args.web or args.hidelogs:
        pass
        # print(msg, val)
    else:
        minesweeper_logger.debug(msg, val)

def log_info(msg, val=None):
    if args.web or args.hidelogs:
        pass
        # print(msg)
    else:
        minesweeper_logger.info(msg)

def autosolve(height, width, mines, solver_method, seed):
    if args.save and rank == 0:
        f = open(args.solver + '_' + str(args.seed) + '_' + str(args.height) + '_' + str(args.width) + '_' + str(args.mines) + '.txt', 'w')
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['width: ' + str(args.width), 'height: ' + str(args.height), 'mines: ' + str(args.mines), 'seed: ' + str(args.seed)])
        writer.writerow(['time_solve_step', 'time_bp_solver', 'time_partial_bp_solver', 'time_custom_reduction'])
    board = msboard.MSBoard(height, width, mines)
    board.init_board(seed)

    my_board = np.zeros((board.board_height, board.board_width))
    # print(len(my_board))
    # print(len(my_board[0]))
    total_time = 0

    for i in range(len(board.info_map)):
        for j in range(len(board.info_map[0])):
            my_board[i][j] = board.info_map[i][j] if board.info_map[i][j] <= 8 else -1

    # failed_to_solve = False
    while board.check_board() == 2:
        return_dict = solver_method(my_board, NUM_THREADS) # NUM_THREADS is unused in distributed
        # if len(return_dict['grids']) == 0:
        #     failed_to_solve = True
        #     break
        if args.save and rank == 0:
            writer.writerow(return_dict['times'])
        total_time += return_dict['times'][0]
        tile = return_dict['grids'][0]
        board.click_field(tile[0], tile[1])
        my_board = np.zeros((board.board_height, board.board_width))
        for i in range(len(board.info_map)):
            for j in range(len(board.info_map[0])):
                my_board[i][j] = board.info_map[i][j] if board.info_map[i][j] <= 8 else -1

        comm.Barrier()
        if rank == 0 and not args.hidelogs:
            board.print_board()
        comm.Barrier()

    # if failed_to_solve or (args.keeptrying and board.check_board() == 0):
    if args.keeptrying and board.check_board() == 0:
        # pass
        autosolve(height, width, mines, solver_method, int(seed+1))
    else:
        if rank == 0:
            if args.save:
                f.close()
                
            if not os.path.exists('total_times.txt'):
                with open("total_times.txt", "w") as totals:
                    writer = csv.writer(totals, delimiter='\t')
                    writer.writerow(['solver', 'threads', 'width', 'height', 'mines', 'seed', 'total_time'])
            totals = open('total_times.txt', 'a')
            writer = csv.writer(totals, delimiter='\t')
            writer.writerow([str(args.solver), str(args.p), str(args.width), str(args.height), str(args.mines), str(args.seed), str(total_time)])
            totals.close()
            print('Finished in ' + str(total_time) + ' seconds')

#######################################
### COMMON METHODS ####################
######################################
def is_unopened(board, index):
    """Takes in a tuple 'index', and board.
    Returns true if tile is unopened"""
    return board[index[0]][index[1]] == -1

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

def is_opened(board, index):
    return not is_unopened(board, index)

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
        var.append(LpVariable('alpha' + str(i), lowBound=0, upBound=1, cat='Integer'))

    if len(constraints) > 0:
        for i in range(len(constraints)):
            var_num, val = constraints[i]
            var[var_num] = val

    constraints_var = []
    for j in range(len(edge_num)):
        constraint = LpVariable('beta' + str(j), lowBound=0, upBound=0, cat='Integer')
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
            elif counts_map[1.0] + counts_map[-1.0] == 1 and row[-1] == 0:
                solved_rows.append(i)  # This row is fully solved
                u = reduce_row(u, row, i, solved_rows)

        if counts_map[-1.0] > 0:
            if counts_map[-1.0] == 1 and row[-1] == 1:
                solved_rows.append(i)
                u = reduce_row(u, row, i, solved_rows, minus_one=True)
            elif counts_map[1.0] + counts_map[-1.0] == 1 and row[-1] == 0:
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
    log_debug("Possible rows \n%s", possible_rows)
    if len(possible_rows) >= num_threads:
        return np.array(possible_rows[-1-num_threads:-1]), selected_b[-1-num_threads:-1]  # Return all the rows.
    else:
        return np.array(possible_rows[:]), selected_b[:]



############################################
### SHARED IMPLEMENTATION #################
###########################################
class myProcess (multiprocessing.Process):
    def __init__(self, threadID, name, counter, linear_mat_reduced, edge_num_reduced, partial_feasible_sol, pos_var, new_pos_var):
        multiprocessing.Process.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.linear_mat_reduced = linear_mat_reduced
        self.edge_num_reduced = edge_num_reduced
        self.partial_feasible_sol = partial_feasible_sol
        self.pos_var = pos_var
        self.new_pos_var = new_pos_var
        # print('linear_mat_reduced: ' + str(linear_mat_reduced))
        # print('edge_num_reduced: ' + str(edge_num_reduced))
        # print('partial_feasible_sol: ' + str(partial_feasible_sol))
        # print('pos_var: ' + str(pos_var))
        # print('new_pos_var: ' + str(new_pos_var))
        self.feas_sol_stride = len(self.linear_mat_reduced[0])
        self.feas_sol_size = self.feas_sol_stride * (len(self.partial_feasible_sol) + len(self.linear_mat_reduced)) * 16
        init_arr = np.empty(self.feas_sol_size)
        init_arr.fill(-1)
        self.feas_sol_serialized = multiprocessing.Array('d', init_arr)
        # self.manager = multiprocessing.Manager()
        # self.feas_sol = self.manager.list()

    def run(self):
        # Get lock to synchronize threads
        threadLock = multiprocessing.Lock()
        threadLock.acquire()
        time_parallel_proc = time.time()
        feas_sol = parallel_solving(self.linear_mat_reduced, self.edge_num_reduced, self.partial_feasible_sol, self.pos_var, self.new_pos_var, self.threadID)
        for j in range(len(feas_sol)):
            for i in range(self.feas_sol_stride):
                # print(str(i) + ', ' + str(j))
                # print('idx: ' + str(j * self.feas_sol_stride + i) + '\n')
                # print(self.feas_sol_size)
                self.feas_sol_serialized[j * self.feas_sol_stride + i] = feas_sol[j][i]
        end_time_parallel_proc = time.time()
        log_info("Proc {} took \n%f".format(self.threadID), end_time_parallel_proc - time_parallel_proc)
        # Free lock to release next thread
        threadLock.release()

    def get_value(self):
        # threadLock.acquire()
        # value = np.reshape(self.feas_sol, (-1, len(self.linear_mat_reduced[0])))
        # print(str(self.feas_sol) + '\n')
        # threadLock.release()
        # return value
        # return self.feas_sol
        feas_sol = []
        max_feas_sols = len(self.partial_feasible_sol) + len(self.linear_mat_reduced)
        for j in range(max_feas_sols):
            row = []
            for i in range(self.feas_sol_stride):
                val = self.feas_sol_serialized[j * self.feas_sol_stride + i]
                if val < 0:
                    # reached end of data, so we break out of both loops
                    break
                row.append(val)
            else:
                # didn't reach end of data, so we use continue to avoid breaking outer loop
                feas_sol.append(np.array(row))
                continue
            if len(feas_sol) == 0:
                feas_sol.append(np.empty(self.feas_sol_stride))
            break
        # print(feas_sol)
        return feas_sol
            # print(self.feas_sol_serialized[i])

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
        # print('linear_mat: ' + str(len(linear_mat_reduced[0])))
        # print('edge_num_reduced: ' + str(len(edge_num_reduced)))
        # print('pos_var: ' + str(len(pos_var)))
        # print('new_pos_var: ' + str(len(new_pos_var)))
        # print('partial_feasible_sol: ' + str(len(partial_feasible_sol)))
        # size = len(self.partial_feasible_sol)
        # print('size: ' + str(size))
        # print('new_pos_var: ' + str(new_pos_var))
        # self.feas_sol = self.manager.list()

    def run(self):
        # Get lock to synchronize threads
        threadLock = threading.Lock()
        threadLock.acquire()
        time_parallel_proc = time.time()
        self.feas_sol = parallel_solving(self.linear_mat_reduced, self.edge_num_reduced, self.partial_feasible_sol, self.pos_var, self.new_pos_var, self.threadID)
        end_time_parallel_proc = time.time()
        log_info("Proc {} took \n%s".format(self.threadID), end_time_parallel_proc - time_parallel_proc)
        # Free lock to release next thread
        threadLock.release()

    def get_value(self):
        # print('feassollen: ' + str(len(self.feas_sol))) + '\n'
        # print('feas_sol: ' + str(len(self.feas_sol[0])))
        return self.feas_sol

def solve_step_shared(board, num_proc):
    time_solve_step = 0
    time_custom_reduction = 0
    time_partial_bp_solver = 0
    time_bp_solver = 0

    start_solve_step = time.time()

    if is_unopened(board, (0, 0)):
        return {'grids':[[0, 0]], 'times':[0, 0, 0, 0]}

    # Prepare the board by getting the linear equations and a mapping of variable to tiles.
    linear_mat, edge_num, pos_var = prepare(board)
    linear_mat_np = np.matrix(linear_mat)  # Convert to np matrix.
    edge_num_np = np.matrix(edge_num)  # Convert to np matrix.

    log_debug("Edge num is: \n%s", edge_num_np)
    # Augment the matrix to do LU decomposition.
    linear_matrix_augmented = np.hstack((linear_mat_np, np.array(edge_num_np).T))

    log_debug("Linear equations augmented matrix is: \n%s", linear_matrix_augmented)
    pl, u = linalg.lu(linear_matrix_augmented, permute_l=True)  # Perform LU decomposition. U is gaussian eliminated.

    log_debug("U matrix of linear equations joined matrix is: \n%s", u)

    # clear_grid_index = choose_clear_grids(u)
    # if len(clear_grid_index) > 0:
    #     log_info("I am sure")
    #     clear_grid_early = []
    #     for i in range(len(clear_grid_index)):
    #         tile_to_open = pos_var[clear_grid_index[i]]
    #         length_of_row = len(board[0])
    #         y_index = tile_to_open / length_of_row
    #         x_index = tile_to_open % length_of_row
    #         clear_grid_early.append([x_index, y_index])
    #     # return clear_grid_early
    #     time_solve_step = time.time() - start_solve_step
    #     return {'grids':clear_grid_early, 'times':[time_solve_step, 0, 0, 0]}
    # else:
    #     log_info("I am guessing")

    

    start_custom_reduction = time.time()
    reduced_u = custom_reduction(u)
    time_custom_reduction = time.time() - start_custom_reduction
    log_debug("Reduced U matrix of lin. eqns joined matrix is: \n%s", reduced_u)
    log_debug("Custom reduction took %s", time_custom_reduction)

    edge_num_reduced = list(reduced_u[:, -1])
    linear_mat_reduced = reduced_u[:, :-1]


    # Select rows that we want to solve as a subproblem in the serial part.
    selected_rows, selected_b = choose_rows(linear_mat_reduced, edge_num_reduced, num_threads = 4)
    selected_rows, new_pos_var = delete_zero_cols(selected_rows, pos_var)

    log_debug("Selected rows \n%s", selected_rows)
    log_debug("New b \n%s", selected_b)
    log_debug("New pos var \n%s", new_pos_var)

    # First bin_programming solve subproblem
    if len(selected_b) != 0:
        log_debug("selected rows \n%s", selected_rows)
        log_debug("selected b \n%s", selected_b)
        start_partial_bp_solver = time.time()
        partial_feasible_sol = solve_binary_program(selected_rows, selected_b)
        time_partial_bp_solver = time.time() - start_partial_bp_solver
        log_debug("Partial feasible sol is \n%s", partial_feasible_sol)
        log_debug("Partial BP solver took %s", time_partial_bp_solver)

        # Imagine that we have distributed
        feas_sol = []
        log_debug("Partial feas sol len \n%s", len(partial_feasible_sol))
    else:
        partial_feasible_sol = []

    if len(partial_feasible_sol) > 1:

        threads = []
        # Create new threads
        for i in range(num_proc):
            # process = multiprocessing.Process(target=parallel_solving, args=(linear_mat_reduced, edge_num_reduced, partial_feasible_sol, pos_var, new_pos_var))
            # threads.append(process)
            threads.append(myProcess(i, "Thread"+str(i), i, linear_mat_reduced, edge_num_reduced, partial_feasible_sol, pos_var, new_pos_var))

        # Start new Threads
        for i in range(num_proc):
            threads[i].start()

        # Wait for all threads to complete
        for t in threads:
            t.join()
        for i in range(num_proc):
            # threads[i].get_value()
            feas_sol += threads[i].get_value()
            # print(feas_sol)

    if len(partial_feasible_sol) <= 1:
        start_bp_solver = time.time()
        serial_feasible_soln = solve_binary_program(linear_mat_reduced, edge_num_reduced)
        time_bp_solver = time.time() - start_bp_solver
        feas_sol = serial_feasible_soln
        log_debug("BP Solver took %s", time_bp_solver)
        log_debug("Length of feasible solution from reduction: \n%s", len(feas_sol))
        log_debug("Length of feasible solution from serial: \n%s", serial_feasible_soln)

    probabilities = np.sum(feas_sol, axis=0)

    grids = []
    # for ind, prob in enumerate(list(probabilities)):
    #     if prob == 0:
    #         tile_to_open = pos_var[ind]
    #         length_of_row = len(board[0])
    #         y_index = tile_to_open / length_of_row
    #         x_index = tile_to_open % length_of_row
    #         grids.append([x_index, y_index])
    if len(probabilities) > 0:
        tile_to_open = pos_var[np.argmin(probabilities)]
        length_of_row = len(board[0])
        y_index = tile_to_open / length_of_row
        x_index = tile_to_open % length_of_row
        grids.append([x_index, y_index])

    time_solve_step = time.time() - start_solve_step
    return {'grids':grids, 'times':[time_solve_step, time_bp_solver, time_partial_bp_solver, time_custom_reduction]}

def solve_shared(board, num_proc):
    global clear_grid
    times = [0, 0, 0, 0]
    if len(clear_grid) == 0:
        output = solve_step_shared(board, num_proc)
        # print(output)
        times = output['times']
        clear_grid = output['grids']
    next_move = clear_grid[-1]
    # print(next_move)
    del clear_grid[-1]
    return [next_move[0], next_move[1]] + times


def parallel_solving(linear_mat_reduced, edge_num_reduced, partial_feasible_sol, pos_var, new_pos_var, threadID):
    feas_sol = []
    for j, sol in enumerate(partial_feasible_sol):
        if j % NUM_THREADS == threadID:
            constraints = [(pos_var.index(new_pos_var[i]), sol[i]) for i in range(len(sol))]
            feas_sol.extend(solve_binary_program(linear_mat_reduced, edge_num_reduced, constraints))
    # print('len: ' + str(len(feas_sol)))
    # print('feas sol: ' + str(np.concatenate(feas_sol))) + '\n'
    # return np.concatenate(feas_sol)
    return feas_sol


#####################################
##### MPI IMPLEMENTATION ###########
###################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
def set_array_for_scatter(arr):
    num_processors = comm.size
    new_arr = [[] for i in range(num_processors)] # This MUST have size comm.size by the end.

    for i, elem in enumerate(arr):
        array_index = i % num_processors
        new_arr[array_index].append(elem)

    return new_arr

def solve_distributed(board):
    global clear_grid_distributed
    times = [0, 0, 0, 0]
    if len(clear_grid_distributed) == 0:
        output = solve_step_distributed(board)
        times = output['times']
        clear_grid_distributed = output['grids']
    next_move = clear_grid_distributed[-1]
    del clear_grid_distributed[-1]
    return [next_move[0], next_move[1]] + times

def solve_step_distributed(board, dummy=None):
    linear_mat_reduced = None
    edge_num_reduced = None
    new_pos_var = None
    pos_var = None

    start_solve_step = time.time()
    time_solve_step = 0

    log_info("I am rank {}".format(rank))
    comm.Barrier()
    log_info("I am rank {}".format(rank))

    if is_unopened(board, (0, 0)):
        return {'grids': [[0, 0]], 'times': [0, 0, 0, 0]}

    # Prepare the board by getting the linear equations and a mapping of variable to tiles.
    if rank == 0:
        linear_mat, edge_num, pos_var = prepare(board)
        linear_mat_np = np.matrix(linear_mat)  # Convert to np matrix.
        edge_num_np = np.matrix(edge_num)  # Convert to np matrix.
    time_partial_bp_solver = 0
    time_custom_reduction = 0
    time_bp_solver = 0


    early_return = False
    clear_grid_early = None

    if rank == 0:
        log_debug("Edge num is: \n%s", edge_num_np)
        # Augment the matrix to do LU decomposition.
        linear_matrix_augmented = np.hstack((linear_mat_np, np.array(edge_num_np).T))
        log_debug("Linear equations augmented matrix is: \n%s", linear_matrix_augmented)
        pl, u = linalg.lu(linear_matrix_augmented, permute_l=True)  # Perform LU decomposition. U is gaussian eliminated.
        log_debug("U matrix of linear equations joined matrix is: \n%s", u)

        # clear_grid_index = choose_clear_grids(u)
        # if len(clear_grid_index) > 0:
        #     log_info("I am sure")
        #     clear_grid_early = []
        #     for i in range(len(clear_grid_index)):
        #         tile_to_open = pos_var[clear_grid_index[i]]
        #         length_of_row = len(board[0])
        #         y_index = tile_to_open / length_of_row
        #         x_index = tile_to_open % length_of_row
        #         clear_grid_early.append([x_index, y_index])
        #     # return clear_grid_early
        #     time_solve_step = time.time() - start_solve_step
        #     early_return = True
        #
        #     # return {'grids': clear_grid_early, 'times': [time_solve_step, 0, 0, 0]}
        # else:
        #     log_info("I am guessing")

        start_custom_reduction = time.time()
        reduced_u = custom_reduction(u)
        time_custom_reduction = time.time() - start_custom_reduction
        log_debug("Reduced U matrix of lin. eqns joined matrix is: \n%s", reduced_u)
        log_debug("Custom reduction took %s", time_custom_reduction)

        edge_num_reduced = list(reduced_u[:, -1])
        linear_mat_reduced = reduced_u[:, :-1]

    early_return = comm.bcast(early_return, root=0)
    clear_grid_early = comm.bcast(clear_grid_early, root=0)
    time_solve_step = comm.bcast(time_solve_step, root=0)
    if early_return:
        return {'grids': clear_grid_early, 'times': [time_solve_step, 0, 0, 0]}




    if rank == 0:
        # Select rows that we want to solve as a subproblem in the serial part.
        selected_rows, selected_b = choose_rows(linear_mat_reduced, edge_num_reduced, num_threads=2)
        selected_rows, new_pos_var = delete_zero_cols(selected_rows, pos_var)

        log_debug("Selected rows \n%s", selected_rows)
        log_debug("New b \n%s", selected_b)
        log_debug("New pos var \n%s", new_pos_var)

    partial_feasible_solution = []
    # First bin_programming solve subproblem
    if rank == 0 and len(selected_b) != 0:
        log_debug("selected rows \n%s", selected_rows)
        log_debug("selected b \n%s", selected_b)
        start_partial_bp_solver = time.time()
        partial_feasible_solution = solve_binary_program(selected_rows, selected_b)
        time_partial_bp_solver = time.time() - start_partial_bp_solver
        log_debug("Partial feasible sol is \n%s", partial_feasible_solution)
        log_debug("Partial BP solver took %s", time_partial_bp_solver)
        log_debug("Partial feas sol len \n%s", len(partial_feasible_solution))


    # partial_feasible_solution. Currently, it's NONE in all ranks except for 0.
    # inside of rank 0 --- 1. its length is more than 2, or its less than 2.
    # IF the length is < 2... i just wanna do serial.
    # If the length > 2 --- then I wanna scatter.
    log_info("I am rank {}".format(rank))
    # if rank == 0:
    log_debug("Partial feasible solution length is, %s", len(partial_feasible_solution))
    # if len(partial_feasible_solution) > 2:
    orig_partial_feasible_solution = comm.bcast(partial_feasible_solution, root=0)
    to_scatter = set_array_for_scatter(partial_feasible_solution)
    log_info("Going to scatter from rank {}".format(rank))
    partial_feasible_solution = comm.scatter(to_scatter, root=0)
    # Broadcast all the data required.
    log_info("Scattered partial_feasible_solution, now broadcasting from rank 0...")
    comm.Barrier()
    log_info("Finished with barrier?")
    linear_mat_reduced = comm.bcast(linear_mat_reduced, root=0)
    edge_num_reduced = comm.bcast(edge_num_reduced, root=0)
    new_pos_var = comm.bcast(new_pos_var, root=0)
    pos_var = comm.bcast(pos_var, root=0)
    time_partial_bp_solver = comm.bcast(time_partial_bp_solver, root=0)
    time_custom_reduction = comm.bcast(time_custom_reduction, root=0)

    log_info("I am rank {}".format(rank))
    comm.Barrier()
    log_info("I am rank {}".format(rank))

    parallel_feasible_solutions = [] # All procs initailize this to be entry.
    log_info("I am rank {}".format(rank))
    parallel_time = -float("inf")
    if partial_feasible_solution != None and len(partial_feasible_solution) > 0 and len(orig_partial_feasible_solution) > 2: # This means that the root sent me something
        for j, sol in enumerate(partial_feasible_solution):

            log_info("Proc {} partial_feasible_sol is {}".format(rank, sol))

            # All processors go through their list of partial_feasible_solutions
            # set timer
            time_parallel_proc = time.time() # All processors time.
            constraints = [(pos_var.index(new_pos_var[i]), sol[i]) for i in range(len(sol))]
            parallel_feasible_solutions.extend(solve_binary_program(linear_mat_reduced, edge_num_reduced, constraints)) # All processors solve.
            end_time_parallel_proc = time.time()

            log_info("Proc {} took {}".format(rank, end_time_parallel_proc - time_parallel_proc)) # All processors log.
            if parallel_time == -float("inf"):
                parallel_time = end_time_parallel_proc - time_parallel_proc
            else:
                parallel_time += end_time_parallel_proc - time_parallel_proc

    min_parallel_time = comm.allreduce(parallel_time, op=MPI.MAX)
    log_info("I am rank {}".format(rank))
    comm.Barrier()
    log_info("I am rank {}".format(rank))

    parallel_feasible_solutions = comm.allreduce(parallel_feasible_solutions, MPI.SUM)
    log_info("Finished gathering... at rank {}, and parallel feasible solutions array is {}".format(rank, parallel_feasible_solutions))


    # if len(partial_feasible_solution) <= 2:


    time_solve_step = 0
    final_feasible_solution = parallel_feasible_solutions
    log_debug("Length of parallel feasible solution: \n%s", len(parallel_feasible_solutions))

    log_info("Original partial feasible soln is {}".format(orig_partial_feasible_solution))
    if len(orig_partial_feasible_solution) <= 2 or final_feasible_solution == []:
        if rank == 0:
            log_info("USING SERIAL SOLVER")
        start_bp_solver = time.time()
        serial_feasible_soln = solve_binary_program(linear_mat_reduced, edge_num_reduced)
        time_bp_solver = time.time() - start_bp_solver
        final_feasible_solution = serial_feasible_soln
        if rank == 0:
            log_info("Serial solver took time {}\n".format(time_bp_solver))
            log_info("Length of serial feasible solution: \n {}".format(serial_feasible_soln))
        else:
            time_solve_step = time_bp_solver

    log_info("Final feasible solution is {}".format(final_feasible_solution))
    probabilities = np.sum(final_feasible_solution, axis=0)
    log_info("Probabilities is {}".format(probabilities))

    if rank == 0 and min_parallel_time > -float("inf"):
        pass
        log_info("Parallel took {}\n".format(min_parallel_time))
    elif min_parallel_time > -float("inf"):
        time_solve_step = min_parallel_time

    grids = []
    # for ind, prob in enumerate(list(probabilities)):
    #     if prob == 0:
    #         tile_to_open = pos_var[ind]
    #         length_of_row = len(board[0])
    #         y_index = tile_to_open / length_of_row
    #         x_index = tile_to_open % length_of_row
    #         grids.append([x_index, y_index])
    # if len(grids) == 0:
    tile_to_open = pos_var[np.argmin(probabilities)]
    length_of_row = len(board[0])
    y_index = tile_to_open / length_of_row
    x_index = tile_to_open % length_of_row
    grids.append([x_index, y_index])


    return {'grids': grids, 'times': [time_solve_step, 0, time_partial_bp_solver, time_custom_reduction]}
    # return [x_index, y_index, 0, 0, 0, 0]






#############################################################
############# WEB FRAMEWORK #################################
#############################################################

app = Flask(__name__, static_folder='static', static_url_path='')


@app.route('/')
def index():
    options = {'autostart': app.config.get('autostart'), 'proc': app.config.get('solver')}
    return render_template('index.html', options=options)

@app.route('/api/solve_next', methods=['POST'])
def solve_next():
    global gameId, clear_grid
    data = request.get_json(force=True, cache=False)
    if (type(data) == list):
        log_debug("data is {}".format(data))
        return
    if gameId != data["gameId"]:
        clear_grid = []
    gameId = data["gameId"]
    solution = []
    if data["procType"] == "serial":
        solution = solve_shared(data["board"], 1)
        solution.append(0)
    elif data["procType"] == "shared":
        solution = solve_shared(data["board"], NUM_THREADS)
        solution.append(1)
    else:
        print("invalid procType")
        return
    solution.append(data["gameId"])
    return jsonify(solution)


if __name__ == '__main__':
    app.debug = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", default=8, type=int, help="Optional: Board height. Default is 8.")
    parser.add_argument("--width", default=8, type=int, help="Optional: Board width. Default is 8.")
    parser.add_argument("--mines", default=10, type=int, help="Optional: Number of mines. Default is 10. ")
    parser.add_argument("--solver", default='serial', type=str, help="Default is shared. Type of solver: serial, shared or distributed")
    parser.add_argument("--web", help="Enable web browser", action="store_true")
    parser.add_argument("--hidelogs", help="Hide logging info", action="store_true")
    parser.add_argument("--deploy", help="Host over network", action="store_true")
    parser.add_argument("--keeptrying", help="Restart with new seed until a game completes", action="store_true")
    parser.add_argument("--save", help="Save performance data", action="store_true")
    parser.add_argument("--autostart", help="Start auto-solve on launch. NOTE: is always true if not using web", action="store_true")
    parser.add_argument("-p", dest="p", default=1, type=int, help="Number of threads, only for shared implementation")
    parser.add_argument("--seed", default=9999, type=int, help="Set random seed.")
    args = parser.parse_args()
    NUM_THREADS = args.p
    app.config['autostart'] = args.autostart
    app.config['solver'] = args.solver

    solver_method = None
    if args.solver == 'distributed':
        solver_method = solve_step_distributed
    elif args.solver == 'shared':
        NUM_THREADS = args.p
        solver_method = solve_step_shared
    elif args.solver == 'serial':
        NUM_THREADS = 1
        solver_method = solve_step_shared
    else:
        raise

    if args.web:
        if args.deploy:
            app.run(host="0.0.0.0", port=80)
        else:
            port = 5000 + random.randint(0, 999)
            url = "http://127.0.0.1:{0}".format(port)
            threading.Timer(0.5, lambda: webbrowser.open(url) ).start()
            app.run(port=port, debug=False)

    else:
        logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt='%d-%m-%Y:%H:%M:%S',
                            level=logging.INFO)
        minesweeper_logger = logging.getLogger("minesweeper_logger")
        logging.getLogger("pulp").setLevel(logging.WARNING)
        logging.getLogger("flask").setLevel(logging.WARNING)
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
        board.init_board()
        autosolve(args.height, args.width, args.mines, solver_method, args.seed)

