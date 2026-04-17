# implement steepest descent and find alphak using newton's method for finding alphak for a function.
import numpy as np
import matplotlib.pyplot as plt


def newton_method(func, grad, x, g, tol=1e-10, max_iter=100):
    alpha = 1.0
    for _ in range(max_iter):
        x_alpha = x - alpha * g

        # Directional derivative of phi(alpha) = f(x - alpha g)
        f_prime = -np.dot(grad(x_alpha), g)

        # approximation of second derivative in alpha
        eps = 1e-8
        x_alpha_eps = x - (alpha + eps) * g
        f_prime_eps = -np.dot(grad(x_alpha_eps), g)
        f_double_prime = (f_prime_eps - f_prime) / eps

        if abs(f_double_prime) < tol:
            break

        alpha_new = alpha - f_prime / f_double_prime
        if abs(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new

    return alpha


def steepest_descent(func, grad, x0, tol=1e-10, max_iter=1000):
    x = np.array(x0, dtype=float)
    path = [x.copy()]

    for _ in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break

        # find alpha using newton's method
        alpha = newton_method(func, grad, x, g)
        x = x - alpha * g
        path.append(x.copy())

    return x, np.array(path)


def func(x):
    return x[0] ** 2 + x[1] ** 2 + 4 * x[0] + 4 * x[1] + 8


def grad(x):
    return np.array([2 * x[0] + 4, 2 * x[1] + 4], dtype=float)


# Initial guess 
x0 = np.array([0.0, 0.0])

# Perform steepest descent
optimal_x, path = steepest_descent(func, grad, x0)
print(f"Optimal x: {optimal_x}")
print(f"Function value at optimal x: {func(optimal_x)}")

# Plot contour 
x1_values = np.linspace(-10, 10, 200)
x2_values = np.linspace(-10, 10, 200)
X1, X2 = np.meshgrid(x1_values, x2_values)
Z = X1**2 + X2**2 + 4 * X1 + 4 * X2 + 8

plt.figure(figsize=(8, 6))
contours = plt.contour(X1, X2, Z, levels=30)
plt.clabel(contours, inline=True, fontsize=8)

plt.plot(path[:, 0], path[:, 1], "o-", color="red", label="Descent Path")
plt.scatter(optimal_x[0], optimal_x[1], color="blue", s=80, label="Optimal Point")

plt.title("Steepest Descent Optimization")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.show()



    
