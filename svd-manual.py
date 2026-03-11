import numpy as np

def manual_svd(matrix):
    try:
        A = np.asanyarray(matrix, dtype=float)
        if A.ndim != 2:
            return 1, "Input must be 2D", None

        # 1. Compute V and S via eigenvalues of A.T @ A
        # Using eigh as A.T @ A is always symmetric
        ata = A.T @ A
        evals, V = np.linalg.eigh(ata)

        # 2. Sort eigenvalues and vectors in descending order
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        V = V[:, idx]

        # 3. Calculate Singular Values (sqrt of eigenvalues)
        # Ensure values aren't negative due to precision issues
        evals[evals < 0] = 0
        S = np.sqrt(evals)

        # 4. Calculate U: u_i = (A @ v_i) / s_i
        # We only compute U for non-zero singular values to avoid div by zero
        U = np.zeros((A.shape[0], len(S)))
        nonzero = S > 1e-10
        U[:, nonzero] = (A @ V[:, nonzero]) / S[nonzero]

        return 0, "Success", (U, S, V.T)

    except np.linalg.LinAlgError:
        return 2, "Eigenvalue computation failed", None
    except Exception as e:
        return 3, f"Unexpected error: {str(e)}", None


def read_matrix_from_user():
    try:
        dims = input("Enter matrix dimensions (n m): ").strip().split()
        if len(dims) != 2:
            return 1, "Please enter exactly two integers for dimensions.", None
        n, m = map(int, dims)
        if n <= 0 or m <= 0:
            return 1, "Dimensions must be positive integers.", None

        print(f"Enter {n} rows with {m} numbers each (space-separated):")
        rows = []
        for i in range(n):
            parts = input(f"Row {i + 1}: ").strip().split()
            if len(parts) != m:
                return 1, f"Row {i + 1} must have exactly {m} numbers.", None
            rows.append(list(map(float, parts)))
        return 0, "Success", rows
    except ValueError:
        return 1, "Invalid number format.", None
    except Exception as e:
        return 3, f"Unexpected error: {str(e)}", None



if __name__ == "__main__":
    code, msg, mat = read_matrix_from_user()
    if code != 0:
        print(f"Error {code}: {msg}")
    else:
        code, msg, result = manual_svd(mat)
        if code == 0:
            U, S, Vh = result
            print("Singular Values:", S)
            print("Reconstructed:\n", (U * S) @ Vh)
        else:
            print(f"Error {code}: {msg}")