#!/bin/bash
mpiexec -n 4 python app.py --solver distributed --width 8 --height 8 --mines 16 -p 4
