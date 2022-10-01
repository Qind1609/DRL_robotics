from mpi4py import MPI
import os

if MPI.COMM_WORLD.Get_rank() == 0:
    if not os.path.exists("./saved_models"):
        os.mkdir("./saved_models")
