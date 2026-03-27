from temfpy import slater
import tenpy
import numpy as np
import matplotlib.pyplot as plt

n = 4

H = -(np.eye(n, k=1) + np.eye(n, k=-1))

C = slater.correlation_matrix(H)
trunc_par = {"chi_max": 100, "svd_min": 1e-6, "degeneracy_tol": 1e-12}

modes = slater.SchmidtModes.from_correlation_matrix(C[0], int(n/2), trunc_par)
schmidt = slater.SchmidtVectors.from_schmidt_modes(modes, trunc_par)
schmidt_R = slater.SchmidtVectors.from_correlation_matrix(C[0], int(n/2)+1, trunc_par)
A = slater.MPSTensorData.from_schmidt_vectors(schmidt, schmidt_R, "left")
#print(A.sometimes_matrix)
#print(A)

mps = slater.C_to_MPS(C[0], trunc_par)