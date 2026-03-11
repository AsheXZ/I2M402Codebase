import numpy as np
import matplotlib.pyplot as plt

n = int(input("Enter n dimensions: "))
A = np.array([list(map(float, input(f"Row {i+1} of A: ").split())) for i in range(n)])
b = np.array(list(map(float, input("Enter vector b: ").split())))
x = np.array(list(map(float, input("Enter initial guess x0: ").split())))
max_iter = int(input("Enter max # of iterations: "))
tol = 1e-6

def f(x):
    return 0.5 * x.T @ A @ x - b.T @ x

def grad(x):
    return A @ x - b

points = [x.copy()]
loss = [f(x)]

for i in range(max_iter):
    g = grad(x)
    if np.linalg.norm(g) < tol:
        break
    alpha = (g.T @ g) / (g.T @ A @ g)
    x = x - alpha * g
    points.append(x.copy())
    loss.append(f(x))
    print(f"Iter {i+1}: x={x}, loss={f(x):.6f}")

points = np.array(points)
print("Solution:", x)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
if n == 2:
    xv = np.linspace(min(points[:,0])-1, max(points[:,0])+1, 200)
    yv = np.linspace(min(points[:,1])-1, max(points[:,1])+1, 200)
    X, Y = np.meshgrid(xv, yv)
    Z = 0.5 * (A[0,0]*X**2 + (A[0,1]+A[1,0])*X*Y + A[1,1]*Y**2) - b[0]*X - b[1]*Y
    plt.contour(X, Y, Z, levels=30)
    plt.plot(points[:,0], points[:,1], 'ro-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Contour Plot with Steepest Descent Path')
else:
    plt.text(0.5, 0.5, "Contour plot only available for n=2", ha='center', va='center')
    plt.axis('off')

plt.subplot(1,2,2)
plt.plot(loss, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Convergence')

plt.tight_layout()
plt.show()