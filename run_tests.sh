#!/bin/bash
# serial
python app.py --width 8 --height 8 --mines 12 --keeptrying --hidelogs
python app.py --width 10 --height 10 --mines 20 --keeptrying --hidelogs
python app.py --width 12 --height 12 --mines 28 --keeptrying --hidelogs
python app.py --width 15 --height 15 --mines 45 --keeptrying --hidelogs
python app.py --width 20 --height 20 --mines 80 --keeptrying --hidelogs
python app.py --width 25 --height 25 --mines 105 --keeptrying --hidelogs

# shared

# mpi
# mpiexec -n 4 python app.py --solver distributed --width 8 --height 8 --mines 16 -p 4
