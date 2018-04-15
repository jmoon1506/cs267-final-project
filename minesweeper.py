from flask import Flask, render_template, request, jsonify
import sys

def solve(board):
    for j in range(len(board)):
        for i in range(len(board[j])):
            if board[j][i] == -1:
                return [i, j]
    return [-1, -1]







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
    # print(data, file=sys.stdout)
    return jsonify(solve(data))

if __name__ == '__main__':
    app.run(debug=True)