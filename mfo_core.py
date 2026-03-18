"""
mfo_core.py
===========
MFO — Mycelium Fungal Optimizer  (Combo B: M1 + M3 + M4)

Contiene SOLO el algoritmo.
No importa nada relacionado con TSP ni con impresión de resultados.
Para usarlo con cualquier problema basta implementar una función evaluar(ruta).

Mecanismos:
    M1 — Spike + período refractario  (Pleurotus djamor, Adamatzky 2018)
    M3 — Memristor: refuerzo α y decaimiento hiperbólico  (P. ostreatus)
    M4 — Umbral θ adaptativo acoplado a W_medio  (Aspergillus niger)
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


# ══════════════════════════════════════════════════════════════════════════════
#  PARÁMETROS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class MFOParams:
    """
    Parámetros del algoritmo con valores por defecto y justificación biológica.
    Todos los valores tienen origen en datos electrofisiológicos experimentales.
    """

    # M3 — Memristor
    alpha: float = (
        0.20  # tasa de refuerzo   · origen: tasa de despolarización 0.05 mV/s
    )
    rho: float = (
        0.05  # tasa de decaimiento · origen: repolarización 0.02 mV/s (2× más lenta)
    )

    # M1 — Spike
    delta: float = 0.001  # umbral mínimo de mejora · origen: variabilidad spike ±10–20%
    k: int = 10  # factor base del refractario · origen: ratio HF 26s / LF 280s

    # M4 — Umbral θ adaptativo
    theta_max: float = 0.03  # umbral máximo · origen: θ > 0.03 → puerta OR en A. niger

    # Pesos W
    W0: float = 0.40  # peso inicial · red en reposo, permite θ bajo al arranque
    W_max: float = 1.00  # saturación memristiva · origen: lazo I-V del hongo

    # Criterios de parada
    epsilon: float = (
        0.005  # umbral de red olvidada · origen: cese de actividad eléctrica
    )
    K: int = 200  # ventana de estancamiento · origen: trenes de spikes 2.5–6 h
    T_max: int = 5000  # límite absoluto de iteraciones
    T_ref_max: int = 200  # techo del refractario (= 2 × K)

    # Agentes
    N: Optional[int] = None  # None → se usa |V| automáticamente

    # Reproducibilidad
    seed: Optional[int] = None


# ══════════════════════════════════════════════════════════════════════════════
#  ESTADO INTERNO
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class MFOEstado:
    """
    Estado mutable del algoritmo durante la ejecución.
    Separado de MFOParams para poder inspeccionar el estado en cualquier punto.
    """

    n: int
    W: list[list[float]]  # pesos memristivos W[i][j], simétrico
    refractario: list[int]  # iteraciones restantes de bloqueo por nodo
    mejor_solucion: Optional[list[int]] = None
    mejor_fitness: float = math.inf
    sin_mejora: int = 0
    historial: list[dict] = field(default_factory=list)

    @classmethod
    def inicializar(cls, n: int, W0: float) -> "MFOEstado":
        """
        FASE 1 — Inicialización.
        W homogéneo = micelio en reposo, sin preferencia previa (V_mem = −80 mV).
        W es simétrico: W[i][j] = W[j][i] siempre.
        """
        W = [[W0 if i != j else 0.0 for j in range(n)] for i in range(n)]
        refractario = [0] * n
        return cls(n=n, W=W, refractario=refractario)

    def W_medio(self) -> float:
        """Media de todos los pesos de aristas. Solo triángulo superior (W simétrico)."""
        total = 0.0
        count = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                total += self.W[i][j]
                count += 1
        return total / count if count > 0 else 0.0

    def decrementar_refractarios(self) -> None:
        """
        Paso 0 del bucle — decrementar al inicio de cada iteración t,
        antes de construir rutas. El bloqueo de t se aplica en t, se reduce en t+1.
        """
        for j in range(self.n):
            if self.refractario[j] > 0:
                self.refractario[j] -= 1


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 2A — θ ADAPTATIVO  (M4)
# ══════════════════════════════════════════════════════════════════════════════


def calcular_theta(estado: MFOEstado, params: MFOParams, t: int) -> float:
    """
    θ(t) = θ_max × (1 − W_medio / W_max)

    W_medio alto (red aprendida) → θ alto → criterio exigente (AND).
    W_medio bajo (red nueva)     → θ bajo → criterio permisivo (OR).

    CORRECCIÓN E1: θ = 0 en t=1 garantiza exploración total en la primera pasada.
    Biológico: primer potencial de acción espontáneo sin criterio previo.
    """
    if t == 1:
        return 0.0
    wm = estado.W_medio()
    return params.theta_max * (1.0 - wm / params.W_max)


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 2B — CONSTRUCCIÓN DE RUTA
# ══════════════════════════════════════════════════════════════════════════════


def construir_ruta(
    estado: MFOEstado, nodo_inicio: int, rng: random.Random
) -> list[int]:
    """
    El agente construye una ruta guiada por W: P(i,j) ∝ W[i][j].
    Normalización solo sobre candidatos disponibles (no visitados, no refractarios).

    CORRECCIÓN A3: P normalizada solo sobre candidatos disponibles.
    CORRECCIÓN A4: si todos bloqueados, modo emergencia → menor W entre no visitados.

    Biológico: la hifa sigue el gradiente de conductancia (tropismo eléctrico).
    """
    n = estado.n
    visitados = [False] * n
    ruta = [nodo_inicio]
    visitados[nodo_inicio] = True

    for _ in range(n - 1):
        i = ruta[-1]

        # Candidatos: no visitados y no refractarios
        candidatos = [
            j for j in range(n) if not visitados[j] and estado.refractario[j] == 0
        ]

        if not candidatos:
            # Modo emergencia: ignorar refractario, elegir menor W disponible.
            # Biológico: hifa toma la vía de menor resistencia ante obstrucción.
            no_visitados = [j for j in range(n) if not visitados[j]]
            j_elegido = min(no_visitados, key=lambda j: estado.W[i][j])
        else:
            pesos = [estado.W[i][j] for j in candidatos]
            suma = sum(pesos)

            if suma == 0:
                j_elegido = rng.choice(candidatos)
            else:
                # Ruleta sesgada por W
                r = rng.uniform(0, suma)
                acumulado = 0.0
                j_elegido = candidatos[-1]
                for idx, j in enumerate(candidatos):
                    acumulado += pesos[idx]
                    if acumulado >= r:
                        j_elegido = j
                        break

        ruta.append(j_elegido)
        visitados[j_elegido] = True

    return ruta


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 2C — DETECCIÓN DE SPIKE  (M1)
# ══════════════════════════════════════════════════════════════════════════════


def detectar_spikes(
    fitnesses: list[float],
    estado: MFOEstado,
    params: MFOParams,
    theta: float,
    t: int,
) -> tuple[list[bool], list[float]]:
    """
    Condición de spike (doble umbral):
        mejora[a] >= delta  Y  mejora[a] >= theta

    CORRECCIÓN E2: en t=1, spike automático para el mejor agente.
    Biológico: primer potencial de acción espontáneo (Slayman et al., 1976).

    CORRECCIÓN A5: varios spikes en la misma iteración → todos refuerzan W,
    pero solo el de mayor mejora actualiza mejor_fitness (gestionado en actualizar_mejor).
    """
    N = len(fitnesses)
    spike = [False] * N
    mejora = [0.0] * N

    if t == 1:
        a_mejor = min(range(N), key=lambda a: fitnesses[a])
        spike[a_mejor] = True
        mejora[a_mejor] = params.delta  # valor mínimo simbólico
        return spike, mejora

    for a in range(N):
        if fitnesses[a] < estado.mejor_fitness:
            m = (estado.mejor_fitness - fitnesses[a]) / estado.mejor_fitness
            mejora[a] = m
            if m >= params.delta and m >= theta:
                spike[a] = True

    return spike, mejora


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 2D — ACTUALIZACIÓN DE PESOS  (M3 — Memristor)
# ══════════════════════════════════════════════════════════════════════════════


def actualizar_pesos(
    estado: MFOEstado,
    params: MFOParams,
    rutas: list[list[int]],
    spike: list[bool],
    mejora: list[float],
) -> None:
    """
    Paso A — Refuerzo (solo agentes con spike):
        W[i][j] += alpha × mejora[a]   (saturado en W_max)

    Paso B — Decaimiento hiperbólico global (TODAS las aristas):
        W[i][j] = W[i][j] / (1 + rho × W[i][j])

    Orden: refuerzo primero, decaimiento después. Nunca al revés.
    W es simétrico: W[i][j] = W[j][i] siempre.

    Biológico: lazo I-V memristivo de P. ostreatus.
    Decaimiento lento para W pequeños, rápido para W grandes (saturación).
    """
    n = estado.n

    # Paso A — Refuerzo (solo spikes)
    for a, tiene_spike in enumerate(spike):
        if not tiene_spike:
            continue
        ruta = rutas[a]
        m = mejora[a]
        for paso in range(len(ruta)):
            i = ruta[paso]
            j = ruta[(paso + 1) % len(ruta)]
            nuevo = min(estado.W[i][j] + params.alpha * m, params.W_max)
            estado.W[i][j] = nuevo
            estado.W[j][i] = nuevo  # CORRECCIÓN A1: simetría

    # Paso B — Decaimiento hiperbólico global
    for i in range(n):
        for j in range(i + 1, n):
            w = estado.W[i][j]
            w_nuevo = w / (1.0 + params.rho * w)
            estado.W[i][j] = w_nuevo
            estado.W[j][i] = w_nuevo  # CORRECCIÓN A1: simetría


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 2E — PERÍODO REFRACTARIO  (M1)
# ══════════════════════════════════════════════════════════════════════════════


def aplicar_refractario(
    estado: MFOEstado,
    params: MFOParams,
    rutas: list[list[int]],
    spike: list[bool],
    mejora: list[float],
) -> None:
    """
    T_ref = MIN( ENTERO(k / mejora[a]),  T_ref_max )

    Inversamente proporcional a la mejora:
      Mejor mejora → T_ref corto  (región prometedora, bloqueo breve)
      Peor mejora  → T_ref largo  (región marginal, forzar exploración)

    CORRECCIÓN E3: cota T_ref_max evita bloqueo casi permanente.
    CORRECCIÓN A2: bloquea el nodo destino j (no la arista).
    Nodo de inicio nunca se bloquea.

    Biológico: spike LF (mayor amplitud) tiene T_ref 10× más largo que HF.
    """
    for a, tiene_spike in enumerate(spike):
        if not tiene_spike:
            continue
        m = mejora[a]
        if m <= 0:
            continue
        t_ref = min(int(params.k / m), params.T_ref_max)
        ruta = rutas[a]
        for j in ruta:
            # MAX: si ya estaba bloqueado más tiempo, conservar el mayor
            estado.refractario[j] = max(estado.refractario[j], t_ref)
        # El nodo de inicio nunca se bloquea
        estado.refractario[ruta[0]] = 0


# ══════════════════════════════════════════════════════════════════════════════
#  ACTUALIZACIÓN DEL MEJOR GLOBAL
# ══════════════════════════════════════════════════════════════════════════════


def actualizar_mejor(
    estado: MFOEstado,
    rutas: list[list[int]],
    fitnesses: list[float],
    spike: list[bool],
    mejora: list[float],
) -> None:
    """
    CORRECCIÓN A5: si varios agentes disparan spike, solo el de mayor mejora
    actualiza el mejor global. Todos refuerzan W (hecho en actualizar_pesos).
    """
    mejor_a = -1
    mejor_m = -1.0
    for a, tiene_spike in enumerate(spike):
        if tiene_spike and mejora[a] > mejor_m:
            mejor_m = mejora[a]
            mejor_a = a

    if mejor_a >= 0:
        estado.mejor_solucion = rutas[mejor_a][:]
        estado.mejor_fitness = fitnesses[mejor_a]
        estado.sin_mejora = 0
    else:
        estado.sin_mejora += 1


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL — mfo()
# ══════════════════════════════════════════════════════════════════════════════


def mfo(
    n: int,
    evaluar: Callable[[list[int]], float],
    params: MFOParams,
) -> dict:
    """
    Punto de entrada del algoritmo MFO.

    Parámetros
    ----------
    n        : número de nodos del grafo del problema
    evaluar  : función f(ruta) -> float que calcula el coste de una ruta.
               El algoritmo minimiza este valor.
               La ruta es una lista de n enteros (0..n-1), sin repetición.
               El coste debe incluir el cierre del ciclo (ruta[−1] → ruta[0]).
    params   : instancia de MFOParams con los parámetros del algoritmo.

    Retorna
    -------
    dict con:
      mejor_solucion : list[int]  — ruta de menor coste encontrada
      mejor_fitness  : float      — coste de esa ruta
      iteraciones    : int        — número de iteraciones ejecutadas
      razon_parada   : str        — cuál de las 3 condiciones terminó
      historial      : list[dict] — registro por iteración
      tiempo_seg     : float      — tiempo de ejecución
    """
    rng = random.Random(params.seed)
    N = params.N if params.N is not None else n

    # ── FASE 1: Inicialización ────────────────────────────────────────────────
    estado = MFOEstado.inicializar(n, params.W0)
    t_inicio = time.perf_counter()
    razon_parada = "T_max alcanzado"

    # ── FASE 2: Bucle principal ───────────────────────────────────────────────
    for t in range(1, params.T_max + 1):
        # Paso 0: decrementar refractarios (inicio de cada iteración)
        estado.decrementar_refractarios()  # CORRECCIÓN A6

        # 2A: umbral θ adaptativo
        theta = calcular_theta(estado, params, t)

        # 2B: construir rutas — agente a empieza en nodo (a % n)
        rutas = [construir_ruta(estado, a % n, rng) for a in range(N)]
        fitnesses = [evaluar(r) for r in rutas]

        # 2C: detectar spikes (doble umbral δ y θ)
        spike, mejora = detectar_spikes(fitnesses, estado, params, theta, t)

        # 2D: actualizar pesos W — refuerzo primero, decaimiento después
        actualizar_pesos(estado, params, rutas, spike, mejora)

        # 2E: aplicar período refractario
        aplicar_refractario(estado, params, rutas, spike, mejora)

        # Actualizar mejor solución global
        actualizar_mejor(estado, rutas, fitnesses, spike, mejora)

        # Registrar estado de la iteración
        wm = estado.W_medio()
        estado.historial.append(
            {
                "t": t,
                "mejor_fitness": estado.mejor_fitness,
                "W_medio": round(wm, 6),
                "theta": round(theta, 6),
                "n_spikes": sum(spike),
                "sin_mejora": estado.sin_mejora,
            }
        )

        # 2F: criterios de parada (triple condición OR)
        if estado.sin_mejora >= params.K:
            razon_parada = f"Estancamiento ({params.K} iters sin mejora)"
            break
        if wm < params.epsilon:
            razon_parada = f"Red olvidó (W_medio={wm:.5f} < ε={params.epsilon})"
            break

    tiempo = time.perf_counter() - t_inicio

    return {
        "mejor_solucion": estado.mejor_solucion,
        "mejor_fitness": estado.mejor_fitness,
        "iteraciones": t,
        "razon_parada": razon_parada,
        "historial": estado.historial,
        "tiempo_seg": round(tiempo, 4),
    }
