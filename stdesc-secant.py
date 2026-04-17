# implement steepest descent and find alphak using secant method for line search.
import numpy as np
import matplotlib.pyplot as plt


def secant_method(func, grad, x, g, tol=1e-10, max_iter=100):
    def phi_prime(alpha):
        # Directional derivative of phi(alpha) = f(x - alpha g)
        return -np.dot(grad(x - alpha * g), g)

    alpha_prev = 0.0
    alpha_curr = 1.0
    f_prev = phi_prime(alpha_prev)
    f_curr = phi_prime(alpha_curr)

    for _ in range(max_iter):
        if abs(f_curr) < tol:
            break

        denom = f_curr - f_prev
        if abs(denom) < tol:
            break

        alpha_next = alpha_curr - f_curr * (alpha_curr - alpha_prev) / denom
        if abs(alpha_next - alpha_curr) < tol:
            alpha_curr = alpha_next
            break

        alpha_prev, alpha_curr = alpha_curr, alpha_next
        f_prev, f_curr = f_curr, phi_prime(alpha_curr)

    # Keep alpha positive so x - alpha * g follows a descent step.
    return alpha_curr if alpha_curr > 0 else 1e-8


def steepest_descent(func, grad, x0, tol=1e-10, max_iter=1000):
    x = np.array(x0, dtype=float)
    path = [x.copy()]

    for _ in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break

        # find alpha using secant method
        alpha = secant_method(func, grad, x, g)
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



    
