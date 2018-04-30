#!/bin/bash
# module load python
# sudo apt-get install python-pip python-dev nginx
# pip install --user gunicorn flask
# pip install --user pulp
ip="$(ip route get 8.8.8.8 | awk '{print $NF; exit}')"
echo "Hosting app at $ip:5000"
python app.py --deploy
