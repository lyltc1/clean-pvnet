import numpy as np
import lib.csrc.edge_refine.build.examples.edge_refine as edge_refine


def edge_refine(R, t):
    edge_refine.py_optimization(R, t)


def modify(A):
    B = edge_refine.modify(A)
    return B

def run_code():
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    t = np.array([[0.0], [0.0], [1.0]], dtype=np.float32)
    # edge_refine.getContour(R, t, "/home/lyl/git/clean-pvnet/data/linemod/cat/cat.ply", "./")


if __name__ == '__main__':
    A = [1., 2., 3., 4.]
    B = modify(A)
    print(B)
