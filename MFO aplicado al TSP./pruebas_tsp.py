"""
pruebas_tsp.py
==============
Suite de pruebas del MFO aplicado al TSP.

Cada prueba es una función independiente que puede ejecutarse por separado.
Para agregar una nueva prueba: agregar una función y llamarla en main().

Uso:
    python pruebas_tsp.py                  # ejecuta todas las pruebas
    python pruebas_tsp.py basica           # solo prueba básica
    python pruebas_tsp.py convergencia     # solo análisis de convergencia
    python pruebas_tsp.py parametros       # solo comparación de parámetros
    python pruebas_tsp.py semillas         # solo estabilidad entre semillas
"""

from __future__ import annotations

import sys

from mfo_core import MFOParams, mfo
from mfo_utils import (
    imprimir_resultado,
    imprimir_historial,
    imprimir_spikes,
    imprimir_resumen,
    comparar_resultados,
)
from tsp import TSPInstance


# ══════════════════════════════════════════════════════════════════════════════
#  PARÁMETROS BASE — punto de partida común para todas las pruebas
# ══════════════════════════════════════════════════════════════════════════════

PARAMS_BASE = MFOParams(
    alpha=0.20,
    rho=0.05,
    delta=0.001,
    k=10,
    theta_max=0.03,
    W0=0.40,
    W_max=1.0,
    epsilon=0.005,
    K=200,
    T_max=5000,
    T_ref_max=200,
    N=50,
    seed=0,
)


# ══════════════════════════════════════════════════════════════════════════════
#  PRUEBA 1 — Básica: verificar que el algoritmo corre y produce resultados
# ══════════════════════════════════════════════════════════════════════════════


def prueba_basica() -> None:
    """
    Ejecuta el MFO sobre una instancia pequeña de 20 ciudades.
    Verifica que el resultado es consistente y muestra el historial.
    """
    print("\n" + "═" * 62)
    print("  PRUEBA 1 — Básica (TSP-20)")
    print("═" * 62)

    instancia = TSPInstance.aleatorio(n=20, seed=42)

    resultado = mfo(
        n=instancia.n,
        evaluar=instancia.evaluar,
        params=PARAMS_BASE,
    )

    imprimir_resultado(resultado, instancia.nombre)
    imprimir_historial(resultado["historial"], cada_n=100)
    imprimir_spikes(resultado["historial"])
    imprimir_resumen(resultado["historial"])

    # Verificación de consistencia
    coste_verificado = instancia.evaluar(resultado["mejor_solucion"])
    assert abs(coste_verificado - resultado["mejor_fitness"]) < 1e-6, (
        f"Error de consistencia: {coste_verificado} ≠ {resultado['mejor_fitness']}"
    )
    print(f"\n  ✓ Consistencia verificada: coste recalculado = {coste_verificado:.4f}")

    # Verificar que la ruta es una permutación válida
    ruta = resultado["mejor_solucion"]
    assert sorted(ruta) == list(range(instancia.n)), (
        "Error: la ruta no es una permutación válida"
    )
    print(f"  ✓ Ruta válida: {instancia.n} ciudades sin repetición")


# ══════════════════════════════════════════════════════════════════════════════
#  PRUEBA 2 — Convergencia: analizar cómo mejora el fitness a lo largo del tiempo
# ══════════════════════════════════════════════════════════════════════════════


def prueba_convergencia() -> None:
    """
    Analiza la dinámica de convergencia del MFO en una instancia de 30 ciudades.
    Muestra cómo evolucionan W_medio, θ y el fitness a lo largo de las iteraciones.
    """
    print("\n" + "═" * 62)
    print("  PRUEBA 2 — Convergencia (TSP-30)")
    print("═" * 62)

    instancia = TSPInstance.aleatorio(n=30, seed=7)

    params = MFOParams(
        alpha=0.20,
        rho=0.05,
        delta=0.001,
        k=10,
        theta_max=0.03,
        W0=0.40,
        W_max=1.0,
        epsilon=0.005,
        K=300,
        T_max=5000,
        T_ref_max=200,
        N=50,
        seed=0,
    )

    resultado = mfo(n=instancia.n, evaluar=instancia.evaluar, params=params)

    imprimir_resultado(resultado, instancia.nombre)

    # Mostrar evolución completa con mayor resolución
    print("\n  Evolución por iteración (cada 100):")
    imprimir_historial(resultado["historial"], cada_n=100)

    imprimir_spikes(resultado["historial"])
    imprimir_resumen(resultado["historial"])

    # Solución greedy como referencia
    greedy_ruta = instancia.solucion_greedy(nodo_inicio=0)
    greedy_coste = instancia.evaluar(greedy_ruta)
    mfo_coste = resultado["mejor_fitness"]
    mejora_vs_greedy = (greedy_coste - mfo_coste) / greedy_coste * 100

    print("\n  Comparación con greedy:")
    print(f"  {'Greedy (vecino más cercano)':35s} {greedy_coste:.4f}")
    print(f"  {'MFO':35s} {mfo_coste:.4f}")
    print(f"  {'Mejora MFO vs greedy':35s} {mejora_vs_greedy:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  PRUEBA 3 — Comparación de parámetros
# ══════════════════════════════════════════════════════════════════════════════


def prueba_parametros() -> None:
    """
    Compara el efecto de distintos valores de parámetros clave.
    Misma instancia, misma semilla, distintas configuraciones.
    """
    print("\n" + "═" * 62)
    print("  PRUEBA 3 — Comparación de parámetros (TSP-20)")
    print("═" * 62)

    instancia = TSPInstance.aleatorio(n=20, seed=42)

    configuraciones = {
        "Base (alpha=0.20, rho=0.05)": MFOParams(
            alpha=0.20,
            rho=0.05,
            delta=0.001,
            k=10,
            theta_max=0.03,
            W0=0.40,
            W_max=1.0,
            epsilon=0.005,
            K=200,
            T_max=5000,
            T_ref_max=200,
            N=50,
            seed=0,
        ),
        "Alpha alto (alpha=0.40)": MFOParams(
            alpha=0.40,
            rho=0.05,
            delta=0.001,
            k=10,
            theta_max=0.03,
            W0=0.40,
            W_max=1.0,
            epsilon=0.005,
            K=200,
            T_max=5000,
            T_ref_max=200,
            N=50,
            seed=0,
        ),
        "Rho alto (rho=0.20)": MFOParams(
            alpha=0.20,
            rho=0.20,
            delta=0.001,
            k=10,
            theta_max=0.03,
            W0=0.40,
            W_max=1.0,
            epsilon=0.005,
            K=200,
            T_max=5000,
            T_ref_max=200,
            N=50,
            seed=0,
        ),
        "Delta alto (delta=0.01)": MFOParams(
            alpha=0.20,
            rho=0.05,
            delta=0.01,
            k=10,
            theta_max=0.03,
            W0=0.40,
            W_max=1.0,
            epsilon=0.005,
            K=200,
            T_max=5000,
            T_ref_max=200,
            N=50,
            seed=0,
        ),
        "Theta alto (theta_max=0.05)": MFOParams(
            alpha=0.20,
            rho=0.05,
            delta=0.001,
            k=10,
            theta_max=0.05,
            W0=0.40,
            W_max=1.0,
            epsilon=0.005,
            K=200,
            T_max=5000,
            T_ref_max=200,
            N=50,
            seed=0,
        ),
    }

    resultados = []
    etiquetas = []

    for nombre, params in configuraciones.items():
        print(f"  Ejecutando: {nombre} ...", end="", flush=True)
        r = mfo(n=instancia.n, evaluar=instancia.evaluar, params=params)
        resultados.append(r)
        etiquetas.append(nombre)
        print(f" fitness={r['mejor_fitness']:.4f}")

    comparar_resultados(resultados, etiquetas)


# ══════════════════════════════════════════════════════════════════════════════
#  PRUEBA 4 — Estabilidad entre semillas
# ══════════════════════════════════════════════════════════════════════════════


def prueba_semillas() -> None:
    """
    Evalúa la estabilidad del algoritmo ejecutándolo 10 veces con semillas distintas.
    Misma instancia, mismos parámetros, distintas semillas aleatorias.
    """
    print("\n" + "═" * 62)
    print("  PRUEBA 4 — Estabilidad entre semillas (TSP-25, 10 ejecuciones)")
    print("═" * 62)

    instancia = TSPInstance.aleatorio(n=25, seed=99)
    n_ejecuciones = 10

    resultados = []
    etiquetas = []

    for semilla in range(n_ejecuciones):
        params = MFOParams(
            alpha=0.20,
            rho=0.05,
            delta=0.001,
            k=10,
            theta_max=0.03,
            W0=0.40,
            W_max=1.0,
            epsilon=0.005,
            K=200,
            T_max=5000,
            T_ref_max=200,
            N=50,
            seed=semilla,
        )
        print(f"  Semilla {semilla:2d} ...", end="", flush=True)
        r = mfo(n=instancia.n, evaluar=instancia.evaluar, params=params)
        resultados.append(r)
        etiquetas.append(f"Semilla {semilla:2d}")
        print(f" fitness={r['mejor_fitness']:.4f}")

    comparar_resultados(resultados, etiquetas)

    # Estadísticas adicionales
    fitnesses = [r["mejor_fitness"] for r in resultados]
    f_min = min(fitnesses)
    f_max = max(fitnesses)
    f_mean = sum(fitnesses) / len(fitnesses)
    f_var = sum((f - f_mean) ** 2 for f in fitnesses) / len(fitnesses)
    f_std = f_var**0.5

    print(f"\n  Estadísticas de fitness ({n_ejecuciones} ejecuciones):")
    print(f"  {'Mínimo':20s} {f_min:.4f}")
    print(f"  {'Máximo':20s} {f_max:.4f}")
    print(f"  {'Media':20s} {f_mean:.4f}")
    print(f"  {'Desv. estándar':20s} {f_std:.4f}")
    print(f"  {'Coef. variación':20s} {f_std / f_mean * 100:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

PRUEBAS = {
    "basica": prueba_basica,
    "convergencia": prueba_convergencia,
    "parametros": prueba_parametros,
    "semillas": prueba_semillas,
}


def main() -> None:
    args = sys.argv[1:]

    if not args:
        # Sin argumentos: ejecutar todas las pruebas
        for nombre, fn in PRUEBAS.items():
            fn()
    else:
        # Ejecutar solo las pruebas especificadas
        for arg in args:
            if arg in PRUEBAS:
                PRUEBAS[arg]()
            else:
                print(f"Prueba desconocida: '{arg}'")
                print(f"Pruebas disponibles: {', '.join(PRUEBAS)}")
                sys.exit(1)


if __name__ == "__main__":
    main()
