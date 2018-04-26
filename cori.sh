#!/bin/bash
module load python
# sudo apt-get install python-pip python-dev nginx
# pip install --user gunicorn flask
pip install --user pulp
python minesweeper.py --deploy
ip="$(ip route get 8.8.8.8 | awk '{print $NF; exit}')"
echo "App hosted at $ip:5000"