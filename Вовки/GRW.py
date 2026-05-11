import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    return np.sum(x**2)

def grey_wolf_optimizer_improved(obj_func, dim, search_agents, max_iter, lower_bound, upper_bound):

    alpha_pos = np.zeros(dim)
    alpha_score = float("inf")

    beta_pos = np.zeros(dim)
    beta_score = float("inf")

    delta_pos = np.zeros(dim)
    delta_score = float("inf")

    positions = np.random.uniform(lower_bound, upper_bound, (search_agents, dim))

    convergence_curve = np.zeros(max_iter)

    for it in range(max_iter):
        for i in range(search_agents):
            positions[i] = np.clip(positions[i], lower_bound, upper_bound)
            fitness = obj_func(positions[i])

            if fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = fitness, positions[i].copy()
            elif fitness < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = fitness, positions[i].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, positions[i].copy()

        a = 2 - it * (2 / max_iter)

        for i in range(search_agents):

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * alpha_pos - positions[i])
            X1 = alpha_pos - A1 * D_alpha

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * beta_pos - positions[i])
            X2 = beta_pos - A2 * D_beta

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * delta_pos - positions[i])
            X3 = delta_pos - A3 * D_delta
            
            positions[i] = (X1 + X2 + X3) / 3
            
        convergence_curve[it] = alpha_score
        
        if (it + 1) % 10 == 0:
            print(f"Ітерація {it + 1}: Поточний найкращий результат = {alpha_score:.6f}")

    return alpha_pos, alpha_score, convergence_curve

if __name__ == "__main__":
    dimensions = 3      
    wolves_count = 20
    iterations = 50     
    lb, ub = -10, 10

    print("Страт")
    best_pos, best_val, curve = grey_wolf_optimizer_improved(
        objective_function, dimensions, wolves_count, iterations, lb, ub
    )

    print("\nРЕЗУЛЬТАТИ")
    print(f"Координати (Альфа): {best_pos}")
    print(f"Значення функції: {best_val:.8f}")

    plt.figure(figsize=(10, 5))
    plt.plot(curve, color='blue', linewidth=2, marker='o', markersize=4)
    plt.title("Графік збіжності алгоритму зграї вовків")
    plt.xlabel("Ітерація")
    plt.ylabel("Найкраще значення цільової функції")
    plt.grid(True)
    plt.show()