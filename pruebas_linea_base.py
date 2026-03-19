"""
pruebas_linea_base.py
=====================
Suite de pruebas de línea de base del MFO con instancias de mayor tamaño.

Objetivo: establecer el comportamiento real del algoritmo antes de
aplicar correcciones, para poder medir el impacto de cada mejora.

Pruebas:
    escala          — rendimiento de n=20 a n=150 con una semilla
    dinamica        — evolución detallada de W_medio, theta y spikes en n=50
    bootstrap       — efecto de W0 sobre la cantidad de spikes en n=50
    semillas_large  — estabilidad en n=50 con 10 semillas

Uso:
    python pruebas_linea_base.py
    python pruebas_linea_base.py escala
    python pruebas_linea_base.py dinamica
    python pruebas_linea_base.py bootstrap
    python pruebas_linea_base.py semillas_large
"""

from __future__ import annotations
import sys
import time

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
#  PARÁMETROS ESCALADOS
# ══════════════════════════════════════════════════════════════════════════════


def params_para(n: int, seed: int = 0) -> MFOParams:
    """
    Parámetros escalados proporcionalmente a n.
    K y T_ref_max crecen con n para dar tiempo suficiente
    de explorar instancias grandes.
    """
    return MFOParams(
        alpha=0.20,
        rho=0.05,
        delta=0.001,
        k=10,
        theta_max=0.03,
        W0=0.40,
        W_max=1.0,
        epsilon=0.005,
        K=n * 10,
        T_max=n * 100,
        T_ref_max=n * 20,
        N=min(n, 40),
        seed=seed,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PRUEBA A — Escala
# ══════════════════════════════════════════════════════════════════════════════


def prueba_escala() -> None:
    """
    Ejecuta el MFO sobre instancias de tamaño creciente con una sola semilla.
    Registra fitness MFO, greedy, ratio, tiempo y spikes.
    Objetivo: ver si el ratio MFO/greedy empeora con n.
    """
    print("\n" + "=" * 70)
    print("  PRUEBA A - Escala de rendimiento (n = 20 a 150, seed=0)")
    print("=" * 70)

    tamanios = [20, 30, 50, 75, 100, 150]

    print(
        f"\n  {'n':>4}  {'N':>3}  {'Tiempo':>7}  {'Iters':>6}  "
        f"{'Spikes':>6}  {'MFO':>9}  {'Greedy':>9}  "
        f"{'MFO/G':>6}  {'MFO/BG':>7}"
    )
    print("  " + "-" * 68)

    tabla = []

    for n in tamanios:
        instancia = TSPInstance.aleatorio(n=n, seed=42)
        params = params_para(n, seed=0)

        t0 = time.perf_counter()
        r = mfo(n=instancia.n, evaluar=instancia.evaluar, params=params)
        t1 = time.perf_counter()

        greedy_0 = instancia.evaluar(instancia.solucion_greedy(0))
        mejor_g = min(
            instancia.evaluar(instancia.solucion_greedy(i)) for i in range(min(5, n))
        )
        spikes = sum(h["n_spikes"] for h in r["historial"])
        ratio_g = r["mejor_fitness"] / greedy_0
        ratio_bg = r["mejor_fitness"] / mejor_g

        print(
            f"  {n:>4}  {params.N:>3}  {t1 - t0:>6.2f}s"
            f"  {r['iteraciones']:>6}  {spikes:>6}"
            f"  {r['mejor_fitness']:>9.1f}  {greedy_0:>9.1f}"
            f"  {ratio_g:>5.2f}x  {ratio_bg:>6.2f}x"
        )
        tabla.append(
            dict(
                n=n,
                fitness=r["mejor_fitness"],
                greedy=greedy_0,
                mejor_greedy=mejor_g,
                spikes=spikes,
                iters=r["iteraciones"],
                tiempo=round(t1 - t0, 2),
            )
        )

    print("  " + "-" * 68)
    print("  MFO/G  = ratio respecto a greedy desde nodo 0")
    print("  MFO/BG = ratio respecto al mejor de 5 arranques greedy")

    print("\n  Tendencia del ratio MFO/Greedy con n:")
    for d in tabla:
        barra = "#" * int((d["fitness"] / d["greedy"]) * 5)
        print(f"  n={d['n']:>3}  {d['fitness'] / d['greedy']:.2f}x  {barra}")

    ratio_ini = tabla[0]["fitness"] / tabla[0]["greedy"]
    ratio_fin = tabla[-1]["fitness"] / tabla[-1]["greedy"]
    if ratio_fin > ratio_ini * 1.2:
        conclusion = (
            "El ratio empeora significativamente con n (bootstrap problem se agrava)"
        )
    elif ratio_fin > ratio_ini:
        conclusion = "El ratio empeora moderadamente con n"
    else:
        conclusion = "El ratio se mantiene estable con n"
    print(f"\n  Conclusion: {conclusion}")


# ══════════════════════════════════════════════════════════════════════════════
#  PRUEBA B — Dinámica interna
# ══════════════════════════════════════════════════════════════════════════════


def prueba_dinamica() -> None:
    """
    Analiza en detalle la dinámica interna del algoritmo en n=50.
    Muestra cómo evolucionan W_medio, theta y el fitness en cada spike.
    Objetivo: entender el ritmo de aprendizaje y el tiempo entre spikes.
    """
    print("\n" + "=" * 70)
    print("  PRUEBA B - Dinamica interna (n=50, seed=0)")
    print("=" * 70)

    n = 50
    instancia = TSPInstance.aleatorio(n=n, seed=42)
    params = params_para(n, seed=0)

    r = mfo(n=instancia.n, evaluar=instancia.evaluar, params=params)

    imprimir_resultado(r, instancia.nombre)
    print("\n  Historial cada 50 iteraciones:")
    imprimir_historial(r["historial"], cada_n=50)
    imprimir_spikes(r["historial"])

    # Intervalos entre spikes
    spike_iters = [h["t"] for h in r["historial"] if h["n_spikes"] > 0]
    if len(spike_iters) > 1:
        intervalos = [
            spike_iters[i + 1] - spike_iters[i] for i in range(len(spike_iters) - 1)
        ]
        print("\n  Intervalos entre spikes consecutivos:")
        for i, iv in enumerate(intervalos):
            print(f"    Spike {i + 1} -> {i + 2}:  {iv} iteraciones de silencio")
        print(f"    Media: {sum(intervalos) / len(intervalos):.1f} iters entre spikes")

    imprimir_resumen(r["historial"])

    h = r["historial"]
    iters_sin = sum(1 for x in h if x["n_spikes"] == 0)
    iters_con = sum(1 for x in h if x["n_spikes"] > 0)
    pct = iters_sin / len(h) * 100

    print("\n  Distribucion del tiempo de computo:")
    print(f"    Iteraciones en silencio (sin spike): {iters_sin}  ({pct:.1f}%)")
    print(f"    Iteraciones con spike:               {iters_con}  ({100 - pct:.1f}%)")
    print(f"    El algoritmo aprende en solo el {100 - pct:.1f}% de las iteraciones")


# ══════════════════════════════════════════════════════════════════════════════
#  PRUEBA C — Bootstrap: efecto de W0
# ══════════════════════════════════════════════════════════════════════════════


def prueba_bootstrap() -> None:
    """
    Varia W0 y mide su efecto sobre spikes y calidad de solucion en n=50.
    Hipotesis: W0 alto -> theta(t=2) bajo -> mas spikes -> mejor solucion.
    Objetivo: confirmar si el bootstrap problem es la causa principal
    del bajo rendimiento y cuanto impacto tiene W0 sobre el resultado.
    """
    print("\n" + "=" * 70)
    print("  PRUEBA C - Bootstrap: efecto de W0 (n=50, seed=0)")
    print("=" * 70)

    n = 50
    instancia = TSPInstance.aleatorio(n=n, seed=42)
    greedy = instancia.evaluar(instancia.solucion_greedy(0))

    W0_valores = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]

    print(f"\n  Greedy de referencia: {greedy:.1f}")
    print(
        f"\n  {'W0':>5}  {'theta(t=2)':>10}  {'Spikes':>6}  "
        f"{'Fitness':>9}  {'MFO/G':>6}  {'Mejora%':>8}"
    )
    print("  " + "-" * 54)

    for W0 in W0_valores:
        params = params_para(n, seed=0)
        params.W0 = W0

        r = mfo(n=instancia.n, evaluar=instancia.evaluar, params=params)
        spikes = sum(h["n_spikes"] for h in r["historial"])
        ratio = r["mejor_fitness"] / greedy
        mejora = (
            (r["historial"][0]["mejor_fitness"] - r["mejor_fitness"])
            / r["historial"][0]["mejor_fitness"]
            * 100
        )
        theta_t2 = r["historial"][1]["theta"] if len(r["historial"]) > 1 else 0.0

        print(
            f"  {W0:>5.2f}  {theta_t2:>10.5f}  {spikes:>6}"
            f"  {r['mejor_fitness']:>9.1f}  {ratio:>5.2f}x  {mejora:>7.2f}%"
        )

    print("  " + "-" * 54)
    print("\n  Interpretacion:")
    print("  W0 alto -> theta(t=2) bajo -> umbral permisivo -> mas spikes")
    print("  Si fitness mejora con W0, el bootstrap problem es la causa principal")


# ══════════════════════════════════════════════════════════════════════════════
#  PRUEBA D — Semillas grandes
# ══════════════════════════════════════════════════════════════════════════════


def prueba_semillas_large() -> None:
    """
    Evalua la estabilidad del MFO en n=50 con 10 semillas distintas.
    Objetivo: establecer la distribucion real de resultados en una instancia
    de tamanio representativo para comparacion futura post-correcciones.
    """
    print("\n" + "=" * 70)
    print("  PRUEBA D - Estabilidad en instancia grande (n=50, 10 semillas)")
    print("=" * 70)

    n = 50
    n_ejecuciones = 10
    instancia = TSPInstance.aleatorio(n=n, seed=42)

    greedy_0 = instancia.evaluar(instancia.solucion_greedy(0))
    mejor_greedy = min(
        instancia.evaluar(instancia.solucion_greedy(i)) for i in range(5)
    )

    print(f"\n  Instancia: {instancia.nombre}")
    print(f"  Greedy (nodo 0):       {greedy_0:.1f}")
    print(f"  Mejor greedy (5 arr.): {mejor_greedy:.1f}")

    resultados = []
    etiquetas = []

    for semilla in range(n_ejecuciones):
        params = params_para(n, seed=semilla)
        print(f"  Semilla {semilla:2d} ...", end="", flush=True)
        r = mfo(n=instancia.n, evaluar=instancia.evaluar, params=params)
        spikes = sum(h["n_spikes"] for h in r["historial"])
        ratio = r["mejor_fitness"] / greedy_0
        print(
            f" fitness={r['mejor_fitness']:8.1f}  spikes={spikes:2d}"
            f"  iters={r['iteraciones']:5d}  ratio={ratio:.2f}x"
        )
        resultados.append(r)
        etiquetas.append(f"Semilla {semilla:2d}")

    comparar_resultados(resultados, etiquetas)

    fitnesses = [r["mejor_fitness"] for r in resultados]
    f_min = min(fitnesses)
    f_max = max(fitnesses)
    f_mean = sum(fitnesses) / len(fitnesses)
    f_std = (sum((f - f_mean) ** 2 for f in fitnesses) / len(fitnesses)) ** 0.5
    spikes_por = [sum(h["n_spikes"] for h in r["historial"]) for r in resultados]

    print(f"\n  Estadisticas de fitness ({n_ejecuciones} ejecuciones):")
    print(f"  {'Minimo':25s} {f_min:.1f}  (ratio vs greedy: {f_min / greedy_0:.2f}x)")
    print(f"  {'Maximo':25s} {f_max:.1f}  (ratio vs greedy: {f_max / greedy_0:.2f}x)")
    print(f"  {'Media':25s} {f_mean:.1f}  (ratio vs greedy: {f_mean / greedy_0:.2f}x)")
    print(f"  {'Desv. estandar':25s} {f_std:.1f}")
    print(f"  {'Coef. variacion':25s} {f_std / f_mean * 100:.2f}%")
    print(
        f"  {'Spikes media':25s} {sum(spikes_por) / len(spikes_por):.1f} spikes/ejecucion"
    )
    print(f"  {'Spikes rango':25s} {min(spikes_por)} - {max(spikes_por)}")

    print("\n  Referencia de calidad:")
    print(f"  {'Mejor MFO':25s} {f_min:.1f}")
    print(f"  {'Greedy':25s} {greedy_0:.1f}")
    print(f"  {'Mejor greedy (5 arr.)':25s} {mejor_greedy:.1f}")
    print(f"  {'Gap media vs greedy':25s} {(f_mean - greedy_0) / greedy_0 * 100:+.1f}%")
    print(f"  {'Gap mejor vs greedy':25s} {(f_min - greedy_0) / greedy_0 * 100:+.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  PRUEBA E — Comparacion de correcciones C1 y C2
# ══════════════════════════════════════════════════════════════════════════════


def prueba_correcciones() -> None:
    """
    Compara las 6 configuraciones de correcciones sobre n=50:
      Base        — sin correcciones (linea de base)
      C1          — warmup activo, W uniforme
      C2          — W heuristico, sin warmup
      C1+C2       — warmup + W heuristico
      C3          — termino eta local (b=2.0), sin C1/C2
      C2+C3       — W heuristico + eta local (mejor combinacion)

    Misma instancia (n=50, seed=42), misma semilla (seed=0).
    """
    print()
    print("=" * 70)
    print("  PRUEBA E - Impacto de correcciones C1, C2 y C3 (n=50, seed=0)")
    print("=" * 70)

    n = 50
    instancia = TSPInstance.aleatorio(n=n, seed=42)
    greedy = instancia.evaluar(instancia.solucion_greedy(0))
    mejor_g = min(instancia.evaluar(instancia.solucion_greedy(i)) for i in range(5))
    W_heur = instancia.W_heuristico(W0=0.40)
    eta = instancia.eta_heuristico()
    warmup = max(10, n // 2)

    print("")
    print("  Greedy (nodo 0):       " + f"{greedy:.1f}")
    print("  Mejor greedy (5 arr.): " + f"{mejor_g:.1f}")
    print("  Warmup C1:             " + f"{warmup} iteraciones")
    print("  Beta C3:               2.0")

    # Construir configuraciones aplicando params escalados
    def hacer_params(warmup_v=0, W_ini=None, beta_v=0.0, eta_v=None):
        p = params_para(n, seed=0)
        p.warmup = warmup_v
        p.W_inicial = W_ini
        p.beta = beta_v
        p.eta = eta_v
        return p

    configuraciones = [
        ("Base", hacer_params()),
        ("C1", hacer_params(warmup_v=warmup)),
        ("C2", hacer_params(W_ini=W_heur)),
        ("C1+C2", hacer_params(warmup_v=warmup, W_ini=W_heur)),
        ("C3", hacer_params(beta_v=2.0, eta_v=eta)),
        ("C2+C3", hacer_params(W_ini=W_heur, beta_v=2.0, eta_v=eta)),
    ]

    print("")
    print(
        "  "
        + "{:<10}  {:>9}  {:>6}  {:>6}  {:>6}  {:>7}".format(
            "Config", "Fitness", "MFO/G", "Spikes", "Iters", "Tiempo"
        )
    )
    print("  " + "-" * 58)

    resultados = []
    etiquetas = []

    for nombre, params in configuraciones:
        r = mfo(n=instancia.n, evaluar=instancia.evaluar, params=params)
        spikes = sum(h["n_spikes"] for h in r["historial"])
        ratio = r["mejor_fitness"] / greedy
        print(
            "  "
            + "{:<10}  {:>9.1f}  {:>5.2f}x  {:>6}  {:>6}  {:>6.2f}s".format(
                nombre,
                r["mejor_fitness"],
                ratio,
                spikes,
                r["iteraciones"],
                r["tiempo_seg"],
            )
        )
        resultados.append(r)
        etiquetas.append(nombre)

    print("  " + "-" * 58)

    base_fitness = resultados[0]["mejor_fitness"]
    print("")
    print("  Mejora respecto a la base:")
    for nombre, r in zip(etiquetas[1:], resultados[1:]):
        mejora = (base_fitness - r["mejor_fitness"]) / base_fitness * 100
        signo = "+" if mejora > 0 else ""
        vs_g = r["mejor_fitness"] / greedy
        marca = " << SUPERA GREEDY" if vs_g < 1.0 else ""
        print(
            "    {:<10}  {}{:.2f}%  ({:.2f}x greedy){}".format(
                nombre, signo, mejora, vs_g, marca
            )
        )

    mejor_idx = min(
        range(len(resultados)), key=lambda i: resultados[i]["mejor_fitness"]
    )
    mejor_r = resultados[mejor_idx]
    print("")
    print("  Detalle de spikes en '" + etiquetas[mejor_idx] + "':")
    imprimir_spikes(mejor_r["historial"])
    imprimir_resumen(mejor_r["historial"])


PRUEBAS = {
    "escala": prueba_escala,
    # "dinamica": prueba_dinamica,
    # "bootstrap": prueba_bootstrap,
    # "semillas_large": prueba_semillas_large,
    # "correcciones": prueba_correcciones,
}


def main() -> None:
    args = sys.argv[1:]
    if not args:
        for fn in PRUEBAS.values():
            fn()
    else:
        for arg in args:
            if arg in PRUEBAS:
                PRUEBAS[arg]()
            else:
                print(f"Prueba desconocida: '{arg}'")
                print(f"Disponibles: {', '.join(PRUEBAS)}")
                sys.exit(1)


if __name__ == "__main__":
    main()
