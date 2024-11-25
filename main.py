import pandas as pd
import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import csv
import os

# --------------------
# 1. Chargement des données
# --------------------
data_file = "assets_data.csv"  # Chemin vers le fichier contenant les données
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Le fichier {data_file} est introuvable. Vérifiez son emplacement.")

# Charger les données depuis le fichier CSV
df = pd.read_csv(data_file)

# Convertir les données en un format compatible avec l'algorithme
assets = df.to_dict("records")

# Contraintes
constraints = {
    "min_cash": 0.1,       # Minimum pour la trésorerie
    "max_metals": 0.1,     # Maximum pour les métaux
    "min_bonds": 0.2,      # Minimum pour les obligations
    "max_stocks": 0.5      # Maximum pour les actions
}

# --------------------
# 2. Représentation
# --------------------
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # Maximiser mu, minimiser sigma^2
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_weight", random.random)  # Poids initiaux aléatoires
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_weight, n=len(assets))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --------------------
# 3. Fonction de fitness
# --------------------
def evaluate(individual):
    weights = np.array(individual)
    weights /= weights.sum()  # Normaliser les poids pour qu'ils somment à 1
    mu = sum(w * a["mu"] for w, a in zip(weights, assets))         # Rendement attendu
    sigma2 = sum(w**2 * a["sigma2"] for w, a in zip(weights, assets))  # Variance
    return mu, sigma2

toolbox.register("evaluate", evaluate)

# --------------------
# 4. Réparation pour respecter les contraintes
# --------------------
def repair(individual):
    weights = np.array(individual)
    weights = np.clip(weights, 0, 1)  # S'assurer que les poids restent entre 0 et 1
    weights /= weights.sum()          # Normaliser pour obtenir une somme égale à 1

    # Appliquer les contraintes pour chaque classe d'actifs
    # Trésorerie
    cash_indices = [i for i, a in enumerate(assets) if a["class"] == "trésorerie"]
    min_cash = constraints["min_cash"]
    if weights[cash_indices].sum() < min_cash:
        weights[cash_indices[0]] += min_cash - weights[cash_indices].sum()

    # Métaux précieux
    metals_indices = [i for i, a in enumerate(assets) if a["class"] == "métaux"]
    max_metals = constraints["max_metals"]
    if weights[metals_indices].sum() > max_metals:
        weights[metals_indices[0]] -= weights[metals_indices].sum() - max_metals

    # Obligations
    bonds_indices = [i for i, a in enumerate(assets) if a["class"] == "obligations"]
    min_bonds = constraints["min_bonds"]
    if weights[bonds_indices].sum() < min_bonds:
        weights[bonds_indices[0]] += min_bonds - weights[bonds_indices].sum()

    # Actions
    stocks_indices = [i for i, a in enumerate(assets) if a["class"] == "actions"]
    max_stocks = constraints["max_stocks"]
    if weights[stocks_indices].sum() > max_stocks:
        weights[stocks_indices[0]] -= weights[stocks_indices].sum() - max_stocks

    individual[:] = weights
    return individual

toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=15.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# --------------------
# 5. Population et Algorithme évolutionnaire
# --------------------
population = toolbox.population(n=100)
hof = tools.HallOfFame(10)  # Meilleurs portefeuilles
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", lambda fits: np.mean([f[0] for f in fits]))
stats.register("min", lambda fits: np.min([f[0] for f in fits]))
stats.register("max", lambda fits: np.max([f[0] for f in fits]))

# Boucle d'évolution
for gen in range(50):  # Nombre de générations
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # Croisement et mutation
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.6:  # Probabilité de croisement
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < 0.3:  # Probabilité de mutation
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Réparation
    for ind in offspring:
        repair(ind)

    # Réévaluation des individus invalides
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Mise à jour de la population
    population[:] = offspring

    # Mise à jour du Hall of Fame
    hof.update(population)

    # Statistiques
    record = stats.compile(population)
    print(f"Gen {gen}: {record}")

# --------------------
# 6. Résultats et visualisation
# --------------------
# Sauvegarder les résultats dans un fichier CSV
with open("pareto_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Poids", "Rendement", "Risque"])
    for ind in hof:
        rendement, risque = evaluate(ind)
        writer.writerow([list(np.round(ind, 3)), rendement, risque])

# Tracer le front de Pareto
risques = []
rendements = []
for ind in hof:
    rendement, risque = evaluate(ind)
    rendements.append(rendement)
    risques.append(risque)

plt.scatter(risques, rendements, color='blue')
plt.xlabel("Risque (Variance)")
plt.ylabel("Rendement attendu")
plt.title("Front de Pareto : Allocation d'actifs")
plt.grid(True)
plt.savefig("pareto_front.png")  # Sauvegarder la figure
plt.show()
