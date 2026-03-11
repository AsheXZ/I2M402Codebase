import numpy as np

def perform_svd(matrix):
    try:
        # 1. Input Validation
        A = np.asanyarray(matrix)
        if A.ndim != 2:
            return 1, "Error: Input must be a 2D matrix.", None

        # 2. Compute SVD
        # U: Left singular vectors, S: Singular values, Vh: Right singular vectors (transposed)
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        
        return 0, "Success", (U, S, Vh)

    except np.linalg.LinAlgError:
        return 2, "Error: SVD computation did not converge.", None
    except Exception as e:
        return 3, f"Error: {str(e)}", None


data = [[1, 2, 7], [3, 4, 9], [5, 8, 6]]
code, message, result = perform_svd(data)

if code == 0:
    U, S, Vh = result
    print("Singular Values:", S)
else:
    print(f"[{code}] {message}")