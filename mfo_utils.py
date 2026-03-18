"""
mfo_utils.py
============
Utilidades de salida y análisis para resultados del MFO.

No contiene lógica del algoritmo ni del problema.
Solo formatea y analiza los resultados que mfo() devuelve.
"""

from __future__ import annotations


# ══════════════════════════════════════════════════════════════════════════════
#  IMPRESIÓN DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════


def imprimir_resultado(resultado: dict, nombre_instancia: str = "") -> None:
    """Imprime un resumen formateado del resultado de mfo()."""
    sep = "─" * 62
    print(f"\n{sep}")
    print("  MFO — Mycelium Fungal Optimizer")
    if nombre_instancia:
        print(f"  Instancia: {nombre_instancia}")
    print(sep)
    print(f"  Mejor coste:     {resultado['mejor_fitness']:.4f}")
    print(f"  Iteraciones:     {resultado['iteraciones']}")
    print(f"  Razón de parada: {resultado['razon_parada']}")
    print(f"  Tiempo:          {resultado['tiempo_seg']} s")
    if resultado.get("mejor_solucion"):
        ruta = resultado["mejor_solucion"]
        ruta_str = " → ".join(str(c) for c in ruta)
        print(f"  Ruta:            {ruta_str} → {ruta[0]}")
    print(sep)


def imprimir_historial(historial: list[dict], cada_n: int = 50) -> None:
    """
    Imprime el historial de iteraciones cada N pasos.
    Siempre muestra t=1 y la última iteración.
    """
    if not historial:
        return

    encabezado = (
        f"{'Iter':>6}  {'Mejor coste':>12}  {'W_medio':>8}  "
        f"{'θ':>8}  {'Spikes':>6}  {'Sin mejora':>10}"
    )
    sep = "─" * len(encabezado)
    print(f"\n{encabezado}")
    print(sep)

    for reg in historial:
        t = reg["t"]
        es_ultimo = t == historial[-1]["t"]
        if t == 1 or t % cada_n == 0 or es_ultimo:
            print(
                f"{t:>6}  {reg['mejor_fitness']:>12.4f}  "
                f"{reg['W_medio']:>8.5f}  {reg['theta']:>8.5f}  "
                f"{reg['n_spikes']:>6}  {reg['sin_mejora']:>10}"
            )


def imprimir_spikes(historial: list[dict]) -> None:
    """Imprime solo las iteraciones donde ocurrió al menos un spike."""
    spikes = [r for r in historial if r["n_spikes"] > 0]
    if not spikes:
        print("\nNo se registraron spikes.")
        return

    print(f"\n  Eventos de spike ({len(spikes)} en total):")
    print(f"  {'Iter':>6}  {'Fitness':>12}  {'Spikes':>6}  {'W_medio':>8}  {'θ':>8}")
    print("  " + "─" * 48)
    for r in spikes:
        print(
            f"  {r['t']:>6}  {r['mejor_fitness']:>12.4f}  "
            f"{r['n_spikes']:>6}  {r['W_medio']:>8.5f}  {r['theta']:>8.5f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════


def resumir_historial(historial: list[dict]) -> dict:
    """
    Calcula estadísticas resumidas sobre el historial de una ejecución.

    Retorna un dict con:
      total_iters     : número total de iteraciones
      total_spikes    : número total de eventos spike
      iters_con_spike : iteraciones donde ocurrió al menos un spike
      mejora_total    : porcentaje de mejora desde t=1 hasta el final
      W_medio_inicial : W_medio en t=1
      W_medio_final   : W_medio en la última iteración
      theta_inicial   : θ en t=1
      theta_final     : θ en la última iteración
    """
    if not historial:
        return {}

    primero = historial[0]
    ultimo = historial[-1]

    mejora_total = 0.0
    if primero["mejor_fitness"] > 0:
        mejora_total = (
            (primero["mejor_fitness"] - ultimo["mejor_fitness"])
            / primero["mejor_fitness"]
            * 100
        )

    return {
        "total_iters": len(historial),
        "total_spikes": sum(r["n_spikes"] for r in historial),
        "iters_con_spike": sum(1 for r in historial if r["n_spikes"] > 0),
        "mejora_total_pct": round(mejora_total, 2),
        "fitness_inicial": primero["mejor_fitness"],
        "fitness_final": ultimo["mejor_fitness"],
        "W_medio_inicial": primero["W_medio"],
        "W_medio_final": ultimo["W_medio"],
        "theta_inicial": primero["theta"],
        "theta_final": ultimo["theta"],
    }


def imprimir_resumen(historial: list[dict]) -> None:
    """Imprime el resumen estadístico del historial."""
    stats = resumir_historial(historial)
    if not stats:
        return
    print("\n  Resumen de la ejecución:")
    print(f"  {'Total iteraciones':30s} {stats['total_iters']}")
    print(
        f"  {'Eventos spike':30s} {stats['total_spikes']}  "
        f"(en {stats['iters_con_spike']} iters)"
    )
    print(
        f"  {'Mejora total':30s} {stats['mejora_total_pct']:.2f}%  "
        f"({stats['fitness_inicial']:.2f} → {stats['fitness_final']:.2f})"
    )
    print(
        f"  {'W_medio':30s} {stats['W_medio_inicial']:.5f} → {stats['W_medio_final']:.5f}"
    )
    print(f"  {'θ':30s} {stats['theta_inicial']:.5f} → {stats['theta_final']:.5f}")


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARACIÓN DE MÚLTIPLES EJECUCIONES
# ══════════════════════════════════════════════════════════════════════════════


def comparar_resultados(
    resultados: list[dict],
    etiquetas: list[str] | None = None,
) -> None:
    """
    Imprime una tabla comparativa de múltiples ejecuciones del MFO.
    Útil para comparar distintas configuraciones de parámetros o semillas.
    """
    if not resultados:
        return

    etiquetas = etiquetas or [f"Ejecución {i + 1}" for i in range(len(resultados))]
    max_label = max(len(e) for e in etiquetas)
    ancho = max(max_label, 12)

    sep = "─" * (ancho + 52)
    print("\n  Comparación de ejecuciones")
    print(f"  {sep}")
    print(
        f"  {'Ejecución':<{ancho}}  {'Fitness':>10}  "
        f"{'Iters':>6}  {'Spikes':>6}  {'Tiempo':>8}  Razón parada"
    )
    print(f"  {sep}")

    fitnesses = [r["mejor_fitness"] for r in resultados]
    mejor_fitness = min(fitnesses)

    for label, r in zip(etiquetas, resultados):
        marca = " ★" if r["mejor_fitness"] == mejor_fitness else "  "
        print(
            f"  {label:<{ancho}}  {r['mejor_fitness']:>10.4f}{marca}"
            f"  {r['iteraciones']:>6}  "
            f"{sum(h['n_spikes'] for h in r['historial']):>6}  "
            f"{r['tiempo_seg']:>7.3f}s  {r['razon_parada']}"
        )
    print(f"  {sep}")
    print(
        f"  {'Mejor':>{ancho + 2}}  {mejor_fitness:.4f}  "
        f"  {'Media':>6}  {sum(fitnesses) / len(fitnesses):.4f}"
    )
