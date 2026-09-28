#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agente de otimização da antena.

Uso:

    python optimize.py compare --budget 6000      # compara os algoritmos
    python optimize.py best --budget 40000        # caça o melhor ganho possível

O modo `compare` dá a cada algoritmo o mesmo orçamento de avaliações e um cache
próprio (em memória), de modo que nenhum se aproveite do trabalho do outro — é a
comparação que a atividade pede.

O modo `best` faz o oposto: usa um único cache compartilhado e persistente, e
encadeia as estratégias em três etapas — exploração ampla por reinícios, refino
exaustivo eixo a eixo, e intensificação em pares de eixos para escapar de
cristas. Como o cache sobrevive entre execuções, rodar de novo continua de onde
parou em vez de recomeçar.
"""

import argparse
import json
import os
import random
import sys

from antenna import Antenna, BudgetExhausted
import algorithms as A

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_FILE = os.path.join(HERE, "best_result.json")


def run_comparison(budget, seed):
    """Roda cada algoritmo isolado, com o mesmo orçamento, e tabula o resultado."""
    contenders = [
        ("Subida da Encosta (íngreme)",
         lambda f, rng: A.random_restart_hill_climb(f, restarts=999, rng=rng, steepest=True)),
        ("Subida da Encosta (escolha-primeiro)",
         lambda f, rng: A.random_restart_hill_climb(f, restarts=999, rng=rng, steepest=False)),
        ("Têmpera Simulada",
         lambda f, rng: A.simulated_annealing(f, rng=rng, t0=8.0, cooling=0.9993, step=45)),
        ("Algoritmo Genético",
         lambda f, rng: A.genetic_algorithm(f, population_size=60, generations=999, rng=rng)),
        ("Descida Coordenada",
         lambda f, rng: A.exhaustive_coordinate_descent(f, A.random_point(rng))),
    ]

    results = []
    for name, run in contenders:
        print(f"\n--- {name} (orçamento {budget}) ---")
        f = Antenna(budget=budget, cache_file=None, verbose=False)
        rng = random.Random(seed)
        try:
            result = run(f, rng)
        except BudgetExhausted:
            result = A.Result(name, f.best_point, f.best_gain, f.evaluations, "orçamento esgotado")
        results.append((name, result, f))
        print(f"    ganho {result.gain:.6f}  |  {f.evaluations} avaliações  |  {result.notes}")
        f.close()

    print("\n" + "=" * 82)
    print(f"{'algoritmo':<38}{'ganho':>14}{'avaliações':>14}{'ganho/1k aval.':>16}")
    print("-" * 82)
    for name, result, f in sorted(results, key=lambda r: r[1].gain, reverse=True):
        efficiency = result.gain / max(1, f.evaluations) * 1000
        print(f"{name:<38}{result.gain:>14.6f}{f.evaluations:>14}{efficiency:>16.3f}")
    print("=" * 82)

    champion = max(results, key=lambda r: r[1].gain)
    print(f"\nMelhor: {champion[0]} -> {champion[1].gain:.6f}")
    print(f"Configuração: {Antenna.describe(champion[1].point)}")
    return champion[1]


def run_best(budget, seed, restarts):
    """Pipeline de três etapas para maximizar o ganho, reaproveitando o cache."""
    rng = random.Random(seed)

    with Antenna(budget=budget, verbose=True) as f:
        # Pontos já conhecidos do cache viram sementes: execuções anteriores
        # não são jogadas fora.
        known = sorted(f.cache.items(), key=lambda kv: kv[1], reverse=True)
        seeds = [point for point, _ in known[:5]]
        if seeds:
            print(f" - semeando com {len(seeds)} melhores pontos do cache")

        try:
            print(f"\n[1/3] Exploração: subida da encosta com {restarts} reinícios aleatórios")
            A.random_restart_hill_climb(f, restarts=restarts, rng=rng, steepest=False, seeds=seeds)
            print(f"      melhor até aqui: {f.best_gain:.6f}  ({f.evaluations} avaliações)")

            print("\n[2/3] Refino: descida coordenada exaustiva a partir do melhor ponto")
            A.exhaustive_coordinate_descent(f, f.best_point)
            print(f"      melhor até aqui: {f.best_gain:.6f}  ({f.evaluations} avaliações)")

            print("\n[3/3] Intensificação: busca em pares de eixos ao redor do melhor ponto")
            A.pair_intensify(f, f.best_point, radius=4)
            print(f"      melhor até aqui: {f.best_gain:.6f}  ({f.evaluations} avaliações)")
        except BudgetExhausted:
            print("\n! orçamento esgotado — reportando o melhor encontrado")

        f.report()
        save_result(f.best_point, f.best_gain, f.evaluations)
        return f.best_point, f.best_gain


def save_result(point, gain, evaluations):
    """Grava o melhor resultado, preservando-o se já houver um melhor no arquivo."""
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, "r", encoding="utf-8") as fh:
            previous = json.load(fh)
        if previous.get("gain", float("-inf")) >= gain:
            print(f" - resultado anterior ({previous['gain']:.6f}) permanece o melhor; arquivo mantido")
            return

    payload = {
        "gain": gain,
        "angles": dict(zip(["phi1", "theta1", "phi2", "theta2", "phi3", "theta3"], point)),
        "evaluations": evaluations,
        "url": "/antenna/simulate?" + "&".join(
            f"{name}={value}" for name, value in
            zip(["phi1", "theta1", "phi2", "theta2", "phi3", "theta3"], point)),
    }
    with open(RESULT_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f" - melhor resultado gravado em {os.path.basename(RESULT_FILE)}")


def main():
    parser = argparse.ArgumentParser(description="Otimização do ganho da antena")
    parser.add_argument("mode", choices=["compare", "best"])
    parser.add_argument("--budget", type=int, default=10000,
                        help="avaliações reais permitidas (por algoritmo no modo compare)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restarts", type=int, default=40,
                        help="reinícios da subida da encosta no modo best")
    args = parser.parse_args()

    if args.mode == "compare":
        run_comparison(args.budget, args.seed)
    else:
        run_best(args.budget, args.seed, args.restarts)


if __name__ == "__main__":
    sys.exit(main())
