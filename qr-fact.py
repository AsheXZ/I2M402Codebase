import numpy as np

def gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R

if __name__ == "__main__":
    r = int(input("Rows: "))
    c = int(input("Cols: "))
    entries = list(map(float, input("Entries (space separated): ").split()))
    matrix = np.array(entries).reshape(r, c)
    Q, R = gram_schmidt(matrix)
    print("\nQ:\n", Q)
    print("\nR:\n", R)
