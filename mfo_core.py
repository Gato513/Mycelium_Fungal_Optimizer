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

    # C1 — Warmup: fase de exploración inicial con θ = 0
    # Biológico: fase de alta actividad eléctrica espontánea antes del primer
    # período refractario significativo. 0 = desactivado (comportamiento original).
    warmup: int = 0  # iters con θ=0 al inicio · sugerido: max(10, n//2)

    # C2 — Inicialización heurística de W
    # Si se proporciona, reemplaza la inicialización uniforme W0.
    # Biológico: hifas crecen hacia gradientes de nutrientes, no a ciegas.
    # Calcular fuera del core: W_inicial[i][j] = W0 / coste[i][j], normalizado.
    W_inicial: Optional[list[list[float]]] = None

    # C3 — Término heurístico local en la construcción (estilo ACO)
    # P(i,j) = W[i][j]^alpha_p × eta[i][j]^beta
    # donde eta[i][j] = 1/coste[i][j] (aristas cortas = mayor atracción local).
    # beta=0 desactiva C3 y reproduce el comportamiento original exactamente.
    # Biológico: la hifa responde tanto a la conductancia histórica (W)
    # como a la concentración local de nutrientes (eta, distancia al objetivo).
    alpha_p: float = 1.0  # exponente de W en la construcción · 1.0 = mismo que antes
    beta: float = 0.0  # exponente de eta · 0.0 = desactivado · sugerido: 2.0
    eta: Optional[list[list[float]]] = None  # matriz 1/coste · None = C3 inactivo

    # alpha_debil: obsoleto, mantenido por compatibilidad, ignorado
    alpha_debil: float = 0.0

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
    def inicializar(cls, n: int, W0: float, W_inicial=None) -> "MFOEstado":
        """
        FASE 1 — Inicialización.

        Sin W_inicial (original): W homogeneo = micelio en reposo (V_mem = -80 mV).
        Con W_inicial (C2): usa la matriz precalculada por el llamador.
            Ejemplo: W_inicial[i][j] = W0 / coste[i][j] sesgado por distancia.
            Biologico: hifas crecen hacia gradientes de nutrientes existentes.

        W es simetrico en ambos casos: W[i][j] = W[j][i] siempre.
        """
        if W_inicial is not None:
            W = [
                [W_inicial[i][j] if i != j else 0.0 for j in range(n)] for i in range(n)
            ]
        else:
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
    C1 — WARMUP: si t <= params.warmup, theta = 0 (exploracion total).
    Biologico: fase de alta actividad electrica espontanea inicial del micelio,
    antes de que el periodo refractario significativo entre en juego.

    Despues del warmup:
    theta(t) = theta_max x (1 - W_medio / W_max)

    W_medio alto (red aprendida) -> theta alto -> criterio exigente (AND).
    W_medio bajo (red nueva)     -> theta bajo -> criterio permisivo (OR).

    CORRECCION E1: theta = 0 en t=1 siempre (independiente del warmup).
    """
    if t <= max(1, params.warmup):
        return 0.0
    wm = estado.W_medio()
    return params.theta_max * (1.0 - wm / params.W_max)


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUE 2B — CONSTRUCCIÓN DE RUTA
# ══════════════════════════════════════════════════════════════════════════════


def construir_ruta(
    estado: MFOEstado,
    nodo_inicio: int,
    rng: random.Random,
    params: Optional["MFOParams"] = None,
) -> list[int]:
    """
    El agente construye una ruta guiada por W y opcionalmente por eta (C3).

    Sin C3 (beta=0 o eta=None):
        P(i,j) proporiconal a W[i][j]

    Con C3 (beta>0 y eta provisto):
        P(i,j) proporcional a W[i][j]^alpha_p × eta[i][j]^beta
        donde eta[i][j] = 1/coste[i][j]
        Biológico: la hifa responde a conductancia histórica (W) y a
        concentración local de nutrientes (eta).

    CORRECCIÓN A3: P normalizada solo sobre candidatos disponibles.
    CORRECCIÓN A4: si todos bloqueados, modo emergencia → menor W.
    """
    n = estado.n
    visitados = [False] * n
    ruta = [nodo_inicio]
    visitados[nodo_inicio] = True

    # Parámetros de C3
    usar_eta = params is not None and params.eta is not None and params.beta > 0
    alpha_p = params.alpha_p if params is not None else 1.0
    beta = params.beta if params is not None else 0.0
    eta = params.eta if usar_eta else None

    for _ in range(n - 1):
        i = ruta[-1]

        # Candidatos: no visitados y no refractarios
        candidatos = [
            j for j in range(n) if not visitados[j] and estado.refractario[j] == 0
        ]

        if not candidatos:
            # Modo emergencia: ignorar refractario.
            # Sin eta: elegir menor W. Con eta: elegir mayor W*eta.
            no_visitados = [j for j in range(n) if not visitados[j]]
            if usar_eta:
                j_elegido = max(
                    no_visitados,
                    key=lambda j: (estado.W[i][j] ** alpha_p) * (eta[i][j] ** beta),
                )
            else:
                j_elegido = min(no_visitados, key=lambda j: estado.W[i][j])
        else:
            if usar_eta:
                # C3: combinar W^alpha_p con eta^beta
                pesos = [
                    (estado.W[i][j] ** alpha_p) * (eta[i][j] ** beta)
                    for j in candidatos
                ]
            else:
                pesos = [estado.W[i][j] for j in candidatos]

            suma = sum(pesos)
            if suma == 0:
                j_elegido = rng.choice(candidatos)
            else:
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

    CORRECCIÓN C3 (si alpha_debil > 0): spike relativo.
    Si ningún agente supera el mejor histórico, el mejor agente de la iteración
    dispara un spike débil con mejora proporcional a su ventaja sobre el promedio:
        mejora_debil = (fitness_medio - fitness_mejor) / fitness_medio * alpha_debil
    Esto permite que W siga diferenciándose entre iteraciones silenciosas.
    El spike débil NO actualiza mejor_fitness (manejado en actualizar_mejor).
    Biológico: P. djamor mantiene actividad eléctrica espontánea basal continua.

    CORRECCIÓN A5: varios spikes → todos refuerzan W, solo el mayor actualiza global.
    """
    N = len(fitnesses)
    spike = [False] * N
    mejora = [0.0] * N

    # t=1: spike automático — primer potencial espontáneo
    if t == 1:
        a_mejor = min(range(N), key=lambda a: fitnesses[a])
        spike[a_mejor] = True
        mejora[a_mejor] = params.delta
        return spike, mejora

    # Spikes reales: agentes que superan el mejor histórico
    hay_spike_real = False
    for a in range(N):
        if fitnesses[a] < estado.mejor_fitness:
            m = (estado.mejor_fitness - fitnesses[a]) / estado.mejor_fitness
            mejora[a] = m
            if m >= params.delta and m >= theta:
                spike[a] = True
                hay_spike_real = True

    # C3 — Spike débil si no hubo ningún spike real y alpha_debil > 0
    if not hay_spike_real and params.alpha_debil > 0:
        a_mejor = min(range(N), key=lambda a: fitnesses[a])
        f_mejor = fitnesses[a_mejor]
        f_medio = sum(fitnesses) / N
        if f_medio > 0 and f_mejor < f_medio:
            # Mejora relativa del mejor agente sobre el promedio de la iteración
            m_relativa = (f_medio - f_mejor) / f_medio * params.alpha_debil
            if m_relativa > 0:
                spike[a_mejor] = True
                mejora[a_mejor] = m_relativa
                # Marcar como spike débil: mejora < delta indica que no actualiza global
                # (la distinción la hace actualizar_mejor comparando con mejor_fitness)

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
        # Solo actualizar el mejor global si el fitness realmente mejoró.
        # Los spikes débiles (C3) refuerzan W pero no actualizan mejor_fitness.
        if fitnesses[mejor_a] < estado.mejor_fitness:
            estado.mejor_solucion = rutas[mejor_a][:]
            estado.mejor_fitness = fitnesses[mejor_a]
            estado.sin_mejora = 0
        else:
            # Spike débil: W se refuerza (en actualizar_pesos) pero el contador
            # de estancamiento sigue avanzando normalmente.
            estado.sin_mejora += 1
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
    n         : número de nodos del grafo del problema
    evaluar   : función f(ruta) -> float que calcula el coste de una ruta.
                El algoritmo minimiza este valor.
                La ruta es una lista de n enteros (0..n-1), sin repetición.
                El coste debe incluir el cierre del ciclo (ruta[−1] → ruta[0]).
    params    : instancia de MFOParams con los parámetros del algoritmo.

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
    estado = MFOEstado.inicializar(n, params.W0, params.W_inicial)
    t_inicio = time.perf_counter()
    razon_parada = "T_max alcanzado"

    # ── FASE 2: Bucle principal ───────────────────────────────────────────────
    for t in range(1, params.T_max + 1):
        # Paso 0: decrementar refractarios (inicio de cada iteración)
        estado.decrementar_refractarios()  # CORRECCIÓN A6

        # 2A: umbral θ adaptativo
        theta = calcular_theta(estado, params, t)

        # 2B: construir rutas — agente a empieza en nodo (a % n)
        rutas = [construir_ruta(estado, a % n, rng, params) for a in range(N)]
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
