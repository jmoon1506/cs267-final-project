from flask import Flask, render_template, request, jsonify
import sys
import numpy as np
from scipy import linalg
np.set_printoptions(threshold=np.nan,linewidth = 1000)
from pulp import *

def solve(board):
    if board[0][0] == -1:
        return  [0, 0]
    linear_mat, edge_num, pos_var = prepare(board)
    P, L, U = linalg.lu(linear_mat)
    U = U.astype(int)
    print linear_mat
    print U
    print edge_num
    mine_list = []
    feas_sol = bin_programming(linear_mat, edge_num, mine_list)
    print len(feas_sol)
    probs = np.sum(feas_sol, axis = 0)
    hit_idx = pos_var[np.argmin(probs)]
    x = len(board[0])
    y_idx = hit_idx/x
    x_idx = hit_idx%x

    return [x_idx, y_idx]

def prepare(board):
    x = len(board[0])
    y = len(board)
    pos_var = []
    edge_num = []
    idx_mat = []
    for i in range(x):
        for j in range(y):
            if board[j][i] > 0:
                var_idx = []
                for offset_x in [-1,0,1]:
                    for offset_y in [-1,0,1]:
                        neighbor_x = i + offset_x
                        neighbor_y = j + offset_y
                        if neighbor_x >= 0 and neighbor_y >= 0 and neighbor_x < x and neighbor_y < y and board[neighbor_y][neighbor_x] == -1:
                            cur_neighbor_idx = neighbor_y*x + neighbor_x
                            var_idx.append(cur_neighbor_idx)
                            if cur_neighbor_idx not in pos_var:
                                pos_var.append(cur_neighbor_idx)
                if len(var_idx) > 0:
                    cur_number_idx = j*x + i
                    edge_num.append(board[cur_number_idx/x][cur_number_idx%x])
                    idx_mat.append([pos_var.index(idx) for idx in var_idx])
    
    linear_mat = np.zeros((len(edge_num),len(pos_var)), dtype = np.int8)
    for i in range(len(edge_num)):
        for j in idx_mat[i]:
            linear_mat[i][j] = 1
    edge_num = np.array(edge_num)
    return linear_mat, edge_num, pos_var

def bin_programming(linear_mat, edge_num, mine_list):
    prob = LpProblem("oneStep", LpMinimize)
    var = []
    for i in range(linear_mat.shape[1]):
        if i in mine_list:
            var.append(LpVariable('a' + str(i),lowBound = 1, upBound = 1, cat='Integer'))
            continue
        var.append(LpVariable('a' + str(i),lowBound = 0, upBound = 1, cat='Integer'))
    constraints_var = []
    for j in range(len(edge_num)):
        constraint = LpVariable('b' + str(j),lowBound = 0, upBound = 0, cat='Integer')
        for k in range(linear_mat.shape[1]):
            constraint += var[k]*linear_mat[j][k]
        constraints_var.append(constraint)
    for j in range(len(edge_num)):
        prob += constraints_var[j] == edge_num[j], "constraint "+str(j+1)
    #mineout = open[('mineout.txt'),'w']
    feas_sol = []
    while True:
        prob.solve()
        if LpStatus[prob.status] == "Optimal":
            sol = []
            for i in range(len(var)):
                sol.append(value(var[i]))
            feas_sol.append(sol)
            #print sol
            #print len([var[i] for i in range(len(var)) if value(var[i]) == 1])
            prob += lpSum([var[i] for i in range(len(var)) if value(var[i]) == 1]) <= len([var[i] for i in range(len(var)) if value(var[i]) == 1]) - 1
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
    #print(data)
    return jsonify(solve(data))

if __name__ == '__main__':
    app.run(debug=True)