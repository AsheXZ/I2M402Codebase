import numpy as np
import matplotlib.pyplot as plt

n = int(input("Enter dimension n for matrix A1: "))
k = int(input("Enter the number of blocks k along one dimension (e.g., 4): "))
N_total = k * n
print(f"The partitioned matrix will be {N_total}x{N_total}")

A1 = np.array([list(map(float, input(f"Row {i+1} of A1: ").split())) for i in range(n)])

# Form the partitioned matrix
I = np.eye(n)
O = np.zeros((n, n))
blocks = [[O for _ in range(k)] for _ in range(k)]
for i in range(k):
    blocks[i][i] = A1
    if i > 0:
        blocks[i][i-1] = -I
    if i < k - 1:
        blocks[i][i+1] = -I

A = np.block(blocks)

print("Partitioned Matrix A:")
print(A)

b = np.array(list(map(float, input(f"Enter vector b (of size {N_total}): ").split())))
x = np.array(list(map(float, input(f"Enter initial guess x0 (of size {N_total}): ").split())))
max_iter = int(input("Enter max # of iterations: "))
tol = float(input("Enter tolerance (e.g., 0.05): "))

def f(x):
    return 0.5 * x.T @ A @ x - b.T @ x

points = [x.copy()]
loss = [f(x)]

# Conjugate Gradient initialization
r = b - A @ x
p = r.copy()

for i in range(max_iter):
    if np.linalg.norm(r) < tol:
        break
        
    Ap = A @ p
    r_dot_r = r.T @ r
    
    alpha = r_dot_r / (p.T @ Ap)
    x = x + alpha * p
    r_new = r - alpha * Ap
    
    points.append(x.copy())
    loss.append(f(x))
    print(f"Iter {i+1}: x={x}, loss={f(x):.6f}")
    
    if np.linalg.norm(r_new) < tol:
        break
        
    beta = (r_new.T @ r_new) / r_dot_r
    p = r_new + beta * p
    r = r_new

points = np.array(points)
print("Solution:", x)

# Visualization
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
if N_total == 2:
    xv = np.linspace(min(points[:,0])-1, max(points[:,0])+1, 200)
    yv = np.linspace(min(points[:,1])-1, max(points[:,1])+1, 200)
    X, Y = np.meshgrid(xv, yv)
    Z = 0.5 * (A[0,0]*X**2 + (A[0,1]+A[1,0])*X*Y + A[1,1]*Y**2) - b[0]*X - b[1]*Y
    plt.contour(X, Y, Z, levels=30)
    plt.plot(points[:,0], points[:,1], 'ro-')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Contour Plot with Conjugate Gradient Path')
else:
    plt.text(0.5, 0.5, f"Contour plot only available for N_total=2\n(Current N_total={N_total})", ha='center', va='center')
    plt.axis('off')

plt.subplot(1,2,2)
plt.plot(loss, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Convergence')

plt.tight_layout()
plt.show()
