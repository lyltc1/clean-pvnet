import numpy as np
import cosypose.lib.optimization.CG.build.examples.edge_refine as edge_refine


A = [1.,2.,3.,4.]
B = edge_refine.modify(A)
print(B)

R = np.array([[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]], dtype=np.float32)
t =np.array([[0.0], [0.0], [1.0]], dtype=np.float32)
edge_refine.getContour(R, t)


