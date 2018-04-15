from flask import Flask, render_template, request, jsonify
import sys, json

app = Flask(__name__, static_folder='static', static_url_path='')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/solve_next', methods=['POST'])
def solve_next():
    data = request.get_json()
    # first_name = json['first_name']
    # last_name = json['last_name']
    # return jsonify(first_name=first_name, last_name=last_name)
    return jsonify(data)

# @app.route('/solve_next', methods=['POST'])
# def solve_next():
#     # jsdata = request.form['javascript_data']
#     data = request.get_json()
#     print(data, file=sys.stdout)
#     # return json.loads(jsdata)[0]
#     # json = request.get_json()
#     # first_name = json['first_name']
#     # last_name = json['last_name']
#     # return jsonify(first_name=first_name, last_name=last_name)
#     return "blah"

if __name__ == '__main__':
    app.run(debug=True)

# def solve_next(board):
#     for j in range(len(board)):
#         for i in range(len(board[j])):
#             if board[j][i] == -1:
#                 # print(str(i) + ' ' + str(j))
#                 return [i, j]
#     return random.random()

