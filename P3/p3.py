import pandas as pd
import numpy as np
import random


#  Esta función genera una población inicial aleatoria de recorridos. Cada recorrido es una permutación aleatoria de los nodos del grafo.
def generar_poblacion_inicial(): # GENERAR POBLACION INICIAL ALEATORIA
    poblacion = []
    for _ in range(n_poblacion):
        recorrido = list(range(n_nodos))
        random.shuffle(recorrido)
        poblacion.append(recorrido)
    return poblacion

#Realiza el cruce de dos padres seleccionados para generar un hijo. El cruce se realiza en un punto aleatorio y combina las partes de los padres para formar un nuevo recorrido.
def crucePadresPoblacion(padre1, padre2): # CRUCE DE PADRES
    punto_cruce = random.randint(1, n_nodos - 1)
    hijo = padre1[:punto_cruce]
    for nodo in padre2:
        if nodo not in hijo:
            hijo.append(nodo)
    return hijo

# Aplica mutación a la población seleccionada. La mutación intercambia aleatoriamente la posición de dos nodos en el recorrido. 
def mutacionSeleccionada(populationSelect): # MUTACION DEL SELECCINADO
    for _ in range(int(n_nodos * mutacion)):
        id1, id2 = random.sample(range(1, n_nodos), 2)
        populationSelect[id1], populationSelect[id2] = populationSelect[id2], populationSelect[id1]
    return populationSelect

#  Selecciona los mejores individuos de la población actual basados en su fitness (distancia total recorrida). Los mejores individuos se eligen mediante ordenamiento de los valores de fitness.
def bestSelect(poblacion, fitness, n_seleccionados): # SELECCIONAMOS A LOS MEJORES INDIVIDUOS DE LA POBLACION
    mejores_indices = np.argsort(fitness)[:n_seleccionados]
    mejores_poblacion = [poblacion[idx] for idx in mejores_indices]
    return mejores_poblacion


def totalDistance(recorrido):
    tDistance = 0
    for i in range(len(recorrido) - 1):
        nodo_actual = recorrido[i]
        nodo_siguiente = recorrido[i + 1]
        tDistance += rutas[nodo_actual][nodo_siguiente]
    return tDistance






# Leer el archivo CSV
grafo = pd.read_csv("grafo_rutas.csv", index_col=0)
print(grafo)
rutas = grafo.to_numpy()
n_nodos = 5 # RUTAS
n_poblacion = 10 # POBLACION
mutacion = 0.3 # PROBABILIDAD DE MUTACION
n_generaciones = 200 # NUMERO DE GENERACIONES
nodo_inicial = 0


poblacion = generar_poblacion_inicial() # GNEREACION DE POBLACION INICIAL

for generacion in range(n_generaciones): # ITERACION DE GENERACIONES 
    fitness = [totalDistance(recorrido) for recorrido in poblacion] # CALCULAMOS LA MEDIDA DE CANTIDAD DE RECORRIDO (FITNESS)
    mejores_poblacion = bestSelect(poblacion, fitness, n_poblacion // 2) # SELECCIONAMOS LOS MEJORES DE LA POBLACION
    newGeneration = [] # CREAMOS UNA NUEVA GENERACION PARA LOS INDIVIDUOS
    while len(newGeneration) < n_poblacion:
        padre1, padre2 = random.sample(mejores_poblacion, 2)
        hijo = crucePadresPoblacion(padre1, padre2)
        newGeneration.append(hijo)
    newGeneration = [mutacionSeleccionada(individuo) for individuo in newGeneration] # APLICAMOS MUTACION EN LA NUEVA GENERACION
    poblacion = newGeneration # REEMPLAZAMOS LA OLD GENERATION POR LA NUEVA GENERACION
fitness = [totalDistance(recorrido) for recorrido in poblacion] # SELECCIONAMOS AL MEJOR INDIVIDO DE LA NUEVA GENERACION
mejor_individuo_idx = np.argmin(fitness)
mejor_recorrido = poblacion[mejor_individuo_idx]
indice_nodo_inicial = mejor_recorrido.index(nodo_inicial)                                           # OBTENEMOS EL MEJOR RECORRIDO
mejor_recorrido = mejor_recorrido[indice_nodo_inicial:] + mejor_recorrido[:indice_nodo_inicial]     # DEL NODO

# Imprimir el mejor recorrido y su distancia total
print("Mejor recorrido:", mejor_recorrido)
print("Distancia total:", totalDistance(mejor_recorrido))
