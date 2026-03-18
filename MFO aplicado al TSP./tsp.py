"""
tsp.py
======
Definición del Problema del Viajante (TSP).

Este módulo no sabe nada del algoritmo MFO.
Solo define la estructura del problema y cómo evaluarlo.
Para usar con MFO, pasar instancia.evaluar como función de coste.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class TSPInstance:
    """
    Instancia del Problema del Viajante (TSP).

    Atributos
    ---------
    n      : número de ciudades
    coste  : matriz n×n de distancias (coste[i][j] = distancia i→j)
    nombre : nombre descriptivo de la instancia
    """
    n:      int
    coste:  list[list[float]]
    nombre: str = "TSP"

    # ── Constructores alternativos ────────────────────────────────────────────

    @classmethod
    def desde_coordenadas(
        cls,
        coords: list[tuple[float, float]],
        nombre: str = "TSP",
    ) -> "TSPInstance":
        """
        Construye la instancia desde una lista de coordenadas (x, y).
        Usa distancia euclídea como coste.
        """
        n = len(coords)
        coste = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = coords[i][0] - coords[j][0]
                    dy = coords[i][1] - coords[j][1]
                    coste[i][j] = math.sqrt(dx * dx + dy * dy)
        return cls(n=n, coste=coste, nombre=nombre)

    @classmethod
    def aleatorio(
        cls,
        n:      int,
        seed:   int = 42,
        escala: float = 100.0,
        nombre: str = "Aleatorio",
    ) -> "TSPInstance":
        """Genera una instancia aleatoria de n ciudades en [0, escala]²."""
        rng    = random.Random(seed)
        coords = [(rng.uniform(0, escala), rng.uniform(0, escala))
                  for _ in range(n)]
        nombre_completo = f"{nombre}-{n}"
        return cls.desde_coordenadas(coords, nombre_completo)

    @classmethod
    def desde_matriz(
        cls,
        coste:  list[list[float]],
        nombre: str = "Matricial",
    ) -> "TSPInstance":
        """Construye la instancia directamente desde una matriz de costes."""
        n = len(coste)
        return cls(n=n, coste=coste, nombre=nombre)

    # ── Evaluación ────────────────────────────────────────────────────────────

    def evaluar(self, ruta: list[int]) -> float:
        """
        Calcula el coste total de una ruta como ciclo cerrado.
        La ruta es una lista de n enteros (0..n-1) sin repetición.
        El coste incluye el regreso desde el último nodo al primero.
        """
        total = 0.0
        n_ruta = len(ruta)
        for i in range(n_ruta):
            total += self.coste[ruta[i]][ruta[(i + 1) % n_ruta]]
        return total

    def evaluar_parcial(self, ruta: list[int]) -> float:
        """
        Calcula el coste de una ruta parcial (sin cierre de ciclo).
        Útil para análisis intermedios.
        """
        total = 0.0
        for i in range(len(ruta) - 1):
            total += self.coste[ruta[i]][ruta[i + 1]]
        return total

    # ── Utilidades ────────────────────────────────────────────────────────────

    def vecinos_mas_cercanos(self, nodo: int, k: int = 5) -> list[int]:
        """Retorna los k vecinos más cercanos a un nodo dado."""
        distancias = [
            (j, self.coste[nodo][j])
            for j in range(self.n)
            if j != nodo
        ]
        distancias.sort(key=lambda x: x[1])
        return [j for j, _ in distancias[:k]]

    def solucion_greedy(self, nodo_inicio: int = 0) -> list[int]:
        """
        Construye una solución greedy (vecino más cercano).
        Útil como cota inferior de referencia o solución inicial.
        """
        visitados = [False] * self.n
        ruta      = [nodo_inicio]
        visitados[nodo_inicio] = True

        for _ in range(self.n - 1):
            actual = ruta[-1]
            mejor_j    = -1
            mejor_dist = math.inf
            for j in range(self.n):
                if not visitados[j] and self.coste[actual][j] < mejor_dist:
                    mejor_dist = self.coste[actual][j]
                    mejor_j    = j
            ruta.append(mejor_j)
            visitados[mejor_j] = True

        return ruta

    def eta_heuristico(self) -> list[list[float]]:
        """
        C3 — Calcula la matriz eta[i][j] = 1/coste[i][j].

        eta representa la 'atracción local' de cada arista: aristas cortas
        tienen mayor eta y atraen más a los agentes en cada paso de construcción.

        Biológico: la hifa responde a la concentración local de nutrientes,
        que es inversamente proporcional a la distancia.

        Retorna la matriz eta lista para pasar a MFOParams.eta.
        Los valores de la diagonal son 0.0.
        """
        n = self.n
        eta = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and self.coste[i][j] > 0:
                    eta[i][j] = 1.0 / self.coste[i][j]
        return eta

    def W_heuristico(self, W0: float, W_max: float = 1.0) -> list[list[float]]:
        """
        C2 — Inicializacion heuristica de W proporcional a 1/coste(i,j).

        W[i][j] = W0 / coste[i][j], normalizado para que W_medio = W0
        y acotado a W_max.

        Biologico: las hifas crecen hacia gradientes de nutrientes conocidos.
        Aristas cortas (bajo coste) reciben mayor conductancia inicial.
        Aristas largas (alto coste) reciben menor conductancia inicial.

        Retorna la matriz W lista para pasar a MFOParams.W_inicial.
        """
        n = self.n
        W = [[0.0] * n for _ in range(n)]

        # Paso 1: W[i][j] = W0 / coste[i][j]
        for i in range(n):
            for j in range(n):
                if i != j and self.coste[i][j] > 0:
                    W[i][j] = W0 / self.coste[i][j]

        # Paso 2: normalizar para que W_medio = W0
        vals = [W[i][j] for i in range(n) for j in range(n) if i != j]
        W_medio_actual = sum(vals) / len(vals) if vals else 1.0
        if W_medio_actual > 0:
            factor = W0 / W_medio_actual
            for i in range(n):
                for j in range(n):
                    W[i][j] = min(W[i][j] * factor, W_max)

        # Forzar simetria y diagonal 0
        for i in range(n):
            W[i][i] = 0.0
            for j in range(i + 1, n):
                v = (W[i][j] + W[j][i]) / 2.0
                W[i][j] = W[j][i] = v

        return W

    def __repr__(self) -> str:
        return f"TSPInstance(nombre={self.nombre!r}, n={self.n})"
