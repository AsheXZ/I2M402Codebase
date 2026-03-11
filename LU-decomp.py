# LU decomposition
def lu_decomposition(matrix):
    """
    Performs LU Decomposition of a given square matrix.
    
    Args:
        matrix (list of lists): NxN Matrix A
        
    Returns:
        tuple: L (Lower triangular matrix), U (Upper triangular matrix)
        
    Raises:
        ValueError: If the matrix is singular (determinant is 0).
    """
    n = len(matrix)
    # Initialize L and U
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        # Upper Triangular
        for j in range(i, n):
            U[i][j] = matrix[i][j]
            for k in range(i):
                U[i][j] -= L[i][k] * U[k][j]
        
        # Lower Triangular
        for j in range(i, n):
            if i == j:
                L[i][i] = 1.0  # Diagonal as 1
            else:
                L[j][i] = matrix[j][i]
                for k in range(i):
                    L[j][i] -= L[j][k] * U[k][i]
                if U[i][i] == 0:
                    raise ValueError("Matrix is singular.")
                L[j][i] /= U[i][i]
    
    return L, U

# solve Ly = b
def forward_substitution(L, b):
    """
    Solves the equation Ly = b using forward substitution.
    
    Args:
        L (list of lists): Lower triangular matrix
        b (list): Vector b
        
    Returns:
        list: Solution vector y
    """
    n = len(L)
    y = [0.0] * n
    
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]
    
    return y

# solve Ux = y

def backward_substitution(U, y):
    """
    Solves the equation Ux = y using backward substitution.
    
    Args:
        U (list of lists): Upper triangular matrix
        y (list): Vector y
        
    Returns:
        list: Solution vector x
    """
    n = len(U)
    x = [0.0] * n
    
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i][j] * x[j]
        if U[i][i] == 0:
            raise ValueError("Matrix is singular.")
        x[i] /= U[i][i]
    
    return x

# Function Driver

def lu_solve(matrix, constants):
    """
    Solves the equation Ax = b using LU Decomposition.
    
    Args:
        matrix (list of lists): NxN Matrix A
        constants (list): Vector b
        
    Returns:
        list: Solution vector x
    """
    L, U = lu_decomposition(matrix)
    y = forward_substitution(L, constants)
    x = backward_substitution(U, y)
    return x

def get_input():
    import sys
    try:
        print("--------------------------------------------")
        n = int(input("Enter the dimension of the matrix (A): "))
        
        if n <= 0:
            print("Error: Dimension must be a positive integer.")
            sys.exit(1)

        print(f"\nEnter coefficients for the {n}x{n} matrix:")
        matrix = []
        for i in range(n):
            while True:
                try:
                    row_str = input(f"Row {i+1}: ")
                    row = [float(x) for x in row_str.split()]
                    if len(row) != n:
                        print(f"Error: Row must contain exactly {n} numbers.")
                        continue
                    matrix.append(row)
                    break
                except ValueError:
                    print("Error: Please enter valid numbers.")

        print("\nEnter the constant terms vector (b):")
        while True:
            try:
                b_str = input(f"Vector b ({n}): ")
                constants = [float(x) for x in b_str.split()]
                if len(constants) != n:
                    print(f"Error: Vector must contain exactly {n} numbers.")
                    continue
                break
            except ValueError:
                print("Error: Please enter valid numbers.")

        return n, matrix, constants

    except KeyboardInterrupt:
        print("\nInput cancelled.")
        sys.exit(0)
        
n, matrix, constants = get_input()
solution = lu_solve(matrix, constants)
print("\nSolution vector x:")
for i, val in enumerate(solution):
    print(f"x[{i}] = {val}")