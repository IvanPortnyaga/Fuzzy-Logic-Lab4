import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

ITEMS = [
    ("Спальний мішок", 2, 50),
    ("Намет", 4, 150),
    ("Ноутбук", 3, 200),
    ("Аптечка", 1, 60),
    ("Запас їжі", 5, 120),
    ("Вода (5л)", 5, 80),
    ("Ліхтарик", 1, 30),
    ("Сонячна панель", 2, 90)
]
NUM_ITEMS = len(ITEMS)
MAX_WEIGHT = 12

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=NUM_ITEMS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalKnapsack(individual):
    weight = 0
    value = 0
    for i in range(NUM_ITEMS):
        if individual[i] == 1:
            weight += ITEMS[i][1]
            value += ITEMS[i][2]
    
    if weight > MAX_WEIGHT:
        return 0,
    return value,

toolbox.register("evaluate", evalKnapsack)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=40, stats=stats, halloffame=hof, verbose=False)

    best_ind = hof[0]
    print("\nОптимальний вміст рюкзака (Еволюційний алгоритм)")
    total_weight = 0
    total_value = 0
    
    for i in range(NUM_ITEMS):
        if best_ind[i] == 1:
            name, w, v = ITEMS[i]
            print(f"[+] {name:<15} (Вага: {w} кг, Цінність: {v})")
            total_weight += w
            total_value += v
            
    print("-" * 45)
    print(f"Загальна вага:     {total_weight} кг (Ліміт: {MAX_WEIGHT} кг)")
    print(f"Загальна цінність: {total_value} умовних одиниць\n")

    max_fitness_values = logbook.select("max")
    mean_fitness_values = logbook.select("avg")

    plt.figure(figsize=(10, 6))
    plt.plot(max_fitness_values, color='red', linewidth=2, label='Найкраща особина')
    plt.plot(mean_fitness_values, color='blue', linestyle='--', label='Середнє значення по популяції')
    plt.title('Динаміка еволюційного пошуку', fontsize=14)
    plt.xlabel('Покоління', fontsize=12)
    plt.ylabel('Цінність зібраних речей', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()