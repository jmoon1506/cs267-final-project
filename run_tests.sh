#!/bin/bash
# serial
python app.py --width 8 --height 8 --mines 11 --keeptrying --hidelogs
python app.py --width 10 --height 10 --mines 17 --keeptrying --hidelogs
python app.py --width 12 --height 12 --mines 24 --keeptrying --hidelogs
python app.py --width 15 --height 15 --mines 38 --keeptrying --hidelogs
python app.py --width 20 --height 20 --mines 67 --keeptrying --hidelogs
python app.py --width 25 --height 25 --mines 88 --keeptrying --hidelogs

# shared
python app.py --width 8 --height 8 --mines 11 --keeptrying --hidelogs --solver shared -p 2
python app.py --width 10 --height 10 --mines 17 --keeptrying --hidelogs --solver shared -p 2
python app.py --width 12 --height 12 --mines 24 --keeptrying --hidelogs --solver shared -p 2
python app.py --width 15 --height 15 --mines 38 --keeptrying --hidelogs --solver shared -p 2
python app.py --width 20 --height 20 --mines 67 --keeptrying --hidelogs --solver shared -p 2
python app.py --width 25 --height 25 --mines 88 --keeptrying --hidelogs --solver shared -p 2

python app.py --width 8 --height 8 --mines 11 --keeptrying --hidelogs --solver shared -p 4
python app.py --width 10 --height 10 --mines 17 --keeptrying --hidelogs --solver shared -p 4
python app.py --width 12 --height 12 --mines 24 --keeptrying --hidelogs --solver shared -p 4
python app.py --width 15 --height 15 --mines 38 --keeptrying --hidelogs --solver shared -p 4
python app.py --width 20 --height 20 --mines 67 --keeptrying --hidelogs --solver shared -p 4
python app.py --width 25 --height 25 --mines 88 --keeptrying --hidelogs --solver shared -p 4

python app.py --width 8 --height 8 --mines 11 --keeptrying --hidelogs --solver shared -p 8
python app.py --width 10 --height 10 --mines 17 --keeptrying --hidelogs --solver shared -p 8
python app.py --width 12 --height 12 --mines 24 --keeptrying --hidelogs --solver shared -p 8
python app.py --width 15 --height 15 --mines 38 --keeptrying --hidelogs --solver shared -p 8
python app.py --width 20 --height 20 --mines 67 --keeptrying --hidelogs --solver shared -p 8
python app.py --width 25 --height 25 --mines 88 --keeptrying --hidelogs --solver shared -p 8

python app.py --width 8 --height 8 --mines 11 --keeptrying --hidelogs --solver shared -p 16
python app.py --width 10 --height 10 --mines 17 --keeptrying --hidelogs --solver shared -p 16
python app.py --width 12 --height 12 --mines 24 --keeptrying --hidelogs --solver shared -p 16
python app.py --width 15 --height 15 --mines 38 --keeptrying --hidelogs --solver shared -p 16
python app.py --width 20 --height 20 --mines 67 --keeptrying --hidelogs --solver shared -p 16
python app.py --width 25 --height 25 --mines 88 --keeptrying --hidelogs --solver shared -p 16


# mpi
mpiexec -n 4 python app.py --width 8 --height 8 --mines 11 --keeptrying --hidelogs --solver distributed -p 2
mpiexec -n 4 python app.py --width 10 --height 10 --mines 17 --keeptrying --hidelogs --solver distributed -p 2
mpiexec -n 4 python app.py --width 12 --height 12 --mines 24 --keeptrying --hidelogs --solver distributed -p 2
mpiexec -n 4 python app.py --width 15 --height 15 --mines 38 --keeptrying --hidelogs --solver distributed -p 2
mpiexec -n 4 python app.py --width 20 --height 20 --mines 67 --keeptrying --hidelogs --solver distributed -p 2
mpiexec -n 4 python app.py --width 25 --height 25 --mines 88 --keeptrying --hidelogs --solver distributed -p 2

mpiexec -n 4 python app.py --width 8 --height 8 --mines 11 --keeptrying --hidelogs --solver distributed -p 4
mpiexec -n 4 python app.py --width 10 --height 10 --mines 17 --keeptrying --hidelogs --solver distributed -p 4
mpiexec -n 4 python app.py --width 12 --height 12 --mines 24 --keeptrying --hidelogs --solver distributed -p 4
mpiexec -n 4 python app.py --width 15 --height 15 --mines 38 --keeptrying --hidelogs --solver distributed -p 4
mpiexec -n 4 python app.py --width 20 --height 20 --mines 67 --keeptrying --hidelogs --solver distributed -p 4
mpiexec -n 4 python app.py --width 25 --height 25 --mines 88 --keeptrying --hidelogs --solver distributed -p 4

mpiexec -n 4 python app.py --width 8 --height 8 --mines 11 --keeptrying --hidelogs --solver distributed -p 8
mpiexec -n 4 python app.py --width 10 --height 10 --mines 17 --keeptrying --hidelogs --solver distributed -p 8
mpiexec -n 4 python app.py --width 12 --height 12 --mines 24 --keeptrying --hidelogs --solver distributed -p 8
mpiexec -n 4 python app.py --width 15 --height 15 --mines 38 --keeptrying --hidelogs --solver distributed -p 8
mpiexec -n 4 python app.py --width 20 --height 20 --mines 67 --keeptrying --hidelogs --solver distributed -p 8
mpiexec -n 4 python app.py --width 25 --height 25 --mines 88 --keeptrying --hidelogs --solver distributed -p 8

mpiexec -n 4 python app.py --width 8 --height 8 --mines 11 --keeptrying --hidelogs --solver distributed -p 16
mpiexec -n 4 python app.py --width 10 --height 10 --mines 17 --keeptrying --hidelogs --solver distributed -p 16
mpiexec -n 4 python app.py --width 12 --height 12 --mines 24 --keeptrying --hidelogs --solver distributed -p 16
mpiexec -n 4 python app.py --width 15 --height 15 --mines 38 --keeptrying --hidelogs --solver distributed -p 16
mpiexec -n 4 python app.py --width 20 --height 20 --mines 67 --keeptrying --hidelogs --solver distributed -p 16
mpiexec -n 4 python app.py --width 25 --height 25 --mines 88 --keeptrying --hidelogs --solver distributed -p 16
