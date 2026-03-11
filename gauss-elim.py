import sys

def get_user_input():
    """
    Handles user input for the matrix size, coefficients, and constant vector.
    Returns:
        n (int): Matrix dimension
        matrix (list of lists): The NxN coefficient matrix
        constants (list): The result vector B
    """
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

def gaussian_elimination(n, matrix, constants):
    """
    Performs Gaussian Elimination with Partial Pivoting.
    
    Args:
        n (int): Dimension
        matrix (list of lists): NxN Matrix A
        constants (list): Vector b
        
    Returns:
        list: Solution vector x
        
    Raises:
        ValueError: If the matrix is singular (determinant is 0).
    """
    # Copy Input
    A = [row[:] for row in matrix]
    b = constants[:]
    
    # Error Threshold
    EPSILON = 1e-10

    # Elimination Phase + Pivot Logic
    for i in range(n):
        
        # Partial Pivoting - find the row with the largest absolute value in the current column
        pivot_row = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > abs(A[pivot_row][i]):
                pivot_row = k
        
        # row swap
        if pivot_row != i:
            A[i], A[pivot_row] = A[pivot_row], A[i]
            b[i], b[pivot_row] = b[pivot_row], b[i]
            
        # Check for singularity - can also be used with == 0, but this is safer for floating point.
        if abs(A[i][i]) < EPSILON: 
            raise ValueError(f"Singular Matrix detected. Pivot at A[{i}][{i}] is effectively zero.")

        # Eliminate entries below the pivot
        for k in range(i + 1, n):
            factor = A[k][i] / A[i][i]
            for j in range(i, n):
                A[k][j] -= factor * A[i][j]
            b[k] -= factor * b[i]

    # Back Substitution Phase
    x = [0.0] * n
    
    for i in range(n - 1, -1, -1):
        sum_ax = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - sum_ax) / A[i][i]

    return x

def main():
    n, matrix, constants = get_user_input()

    try:
        solution = gaussian_elimination(n, matrix, constants)
        
        print("\nSolution")
        for i in range(n):
            print(f"x{i+1} = {solution[i]:.4f}")
            
    except ValueError as e:
        print(f"\nMath Error: {e}")
    except Exception as e:
        print(f"\nUnexpected Error: {e}")

# Driver

if __name__ == "__main__": 
    main()