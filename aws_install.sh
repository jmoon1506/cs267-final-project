#!/bin/bash
sudo apt-get update && sudo apt-get -y upgrade
sudo apt-get install python python-pip mpich
git clone https://github.com/jmoon1506/cs267-final-project
cd cs267-final-project
pip install -r requirements.txt
pip install mpi4py
