from deap import creator, base, tools, algorithms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# FUNCION DE EVALUACION
def evaluarRecorrido(individuo):
    distancia_total = 0
    for i in range(n-1):
        recorrido1 = individuo[i]
        recorrido2 = individuo[i+1]
        distancia = grafo[recorrido1][recorrido2]
        distancia_total = distancia_total + distancia
    return distancia_total,
  

# LEEMOS EL ARCHIVO DE GRAFO_RUTAS
grafo = pd.read_csv("grafo_rutas.csv", index_col=0)
print(grafo)

# CONVERTIMOS TODO A DATOS NUMPY
grafo = grafo.to_numpy()

# VARIABLES 
n = 5
toolbox = base.Toolbox()


# MODELO
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("EstrIndividuo", list, fitness=creator.FitnessMin)
toolbox.register("Genes", np.random.permutation, n)
toolbox.register("Individuos", tools.initIterate, creator.EstrIndividuo, toolbox.Genes)
toolbox.register("Poblacion", tools.initRepeat, list, toolbox.Individuos)
pop = toolbox.Poblacion(n=10)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=2)


toolbox.register("evaluate", evaluarRecorrido )

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('mean', np.mean)
stats.register('min', np.min)
stats.register('max', np.max)
hof = tools.HallOfFame(1)
result, log = algorithms.eaSimple(pop,toolbox,cxpb=0.8,mutpb=0.1,stats=stats,ngen=30,halloffame=hof,verbose=True)



print("Mejor Ruta Recorrida:", hof)
mejor_recorrido = list(hof[0])
distancia_recorrida =0 
for i in range(len(mejor_recorrido)- 1):
  distancia_recorrida = distancia_recorrida + rutas[mejor_recorrido[i]][mejor_recorrido[i+1]]
print("Distancia total de la ruta: ", distancia_recorrida)