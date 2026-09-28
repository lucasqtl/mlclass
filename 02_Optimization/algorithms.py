#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algoritmos de busca por melhoria iterativa para a otimização da antena.

Implementados aqui:

  - Subida da Encosta (Hill Climbing), nas variantes:
      * subida mais íngreme (steepest ascent), avaliando toda a vizinhança;
      * escolha-primeiro (first-choice), aceitando o primeiro vizinho melhor;
      * com reinícios aleatórios (random restart);
      * grosso-para-fino, que começa com passos largos e os reduz ao estagnar.
  - Têmpera Simulada (Simulated Annealing), com critério de Metropolis.
  - Algoritmo Genético, com seleção por torneio, cruzamento e mutação cíclicos.
  - Descida coordenada exaustiva, usada como linha de base determinística: como
    cada eixo só tem 360 valores, dá para varrer um eixo inteiro por vez.

Todo o espaço é cíclico (os ângulos são tomados módulo 360), então os vizinhos
podem "dar a volta" livremente — é por isso que as diferenças entre ângulos usam
`signed_delta`, que devolve o menor caminho no círculo.
"""

import math
import random

from antenna import BudgetExhausted, DIM, normalize


def signed_delta(a, b):
    """Menor diferença angular de a para b, em (-180, 180]."""
    return ((b - a + 180) % 360) - 180


def perturb(point, axis, offset):
    new = list(point)
    new[axis] = (new[axis] + offset) % 360
    return tuple(new)


def random_point(rng):
    return tuple(rng.randrange(360) for _ in range(DIM))


class Result:
    def __init__(self, name, point, gain, evals, notes=""):
        self.name = name
        self.point = point
        self.gain = gain
        self.evals = evals
        self.notes = notes

    def __repr__(self):
        return f"<{self.name}: {self.gain:.6f} em {self.evals} avaliações>"


# --------------------------------------------------------------------------
# Subida da Encosta
# --------------------------------------------------------------------------

DEFAULT_STEPS = (64, 32, 16, 8, 4, 2, 1)


def hill_climb(f, start, steps=DEFAULT_STEPS, steepest=True, rng=None):
    """Sobe a encosta a partir de `start`, refinando o passo ao estagnar.

    Com `steepest=True` avalia todos os vizinhos e salta para o melhor; com
    `steepest=False` (escolha-primeiro) salta assim que encontra uma melhora,
    o que gasta menos avaliações por passo mas costuma exigir mais passos.
    """
    rng = rng or random
    current = normalize(start)
    current_gain = f(current)

    for step in steps:
        while True:
            moves = [(axis, sign * step) for axis in range(DIM) for sign in (1, -1)]
            if not steepest:
                rng.shuffle(moves)

            best_move, best_gain = None, current_gain
            for axis, offset in moves:
                gain = f(perturb(current, axis, offset))
                if gain > best_gain:
                    best_move, best_gain = (axis, offset), gain
                    if not steepest:
                        break

            if best_move is None:
                break
            current = perturb(current, *best_move)
            current_gain = best_gain

    return current, current_gain


def pair_intensify(f, point, radius=3):
    """Busca local em pares de eixos, para escapar de cristas (ridges).

    A subida coordenada trava quando o ótimo fica numa crista diagonal: nenhum
    movimento em um único eixo melhora, mas mover dois eixos juntos melhora.
    Aqui todos os 15 pares de eixos são varridos com deslocamentos pequenos.
    """
    current = normalize(point)
    current_gain = f(current)

    improved = True
    while improved:
        improved = False
        for a in range(DIM):
            for b in range(a + 1, DIM):
                for da in range(-radius, radius + 1):
                    for db in range(-radius, radius + 1):
                        if da == 0 and db == 0:
                            continue
                        candidate = perturb(perturb(current, a, da), b, db)
                        gain = f(candidate)
                        if gain > current_gain:
                            current, current_gain = candidate, gain
                            improved = True
    return current, current_gain


def random_restart_hill_climb(f, restarts=20, rng=None, steepest=True, seeds=()):
    """Reinicia a subida de encosta em vários pontos e guarda o melhor topo.

    `seeds` permite começar por pontos já conhecidos como bons antes de partir
    para reinícios uniformemente aleatórios.
    """
    rng = rng or random
    best_point, best_gain = None, float("-inf")
    optima = []

    starts = [normalize(s) for s in seeds]
    starts += [random_point(rng) for _ in range(restarts)]

    for start in starts:
        try:
            point, gain = hill_climb(f, start, steepest=steepest, rng=rng)
        except BudgetExhausted:
            break
        optima.append((gain, point))
        if gain > best_gain:
            best_point, best_gain = point, gain

    notes = f"{len(optima)} reinícios concluídos"
    if optima:
        gains = sorted((g for g, _ in optima), reverse=True)
        notes += f"; ótimos locais de {gains[0]:.4f} a {gains[-1]:.4f}"
    return Result("hill_climbing", best_point, best_gain, f.evaluations, notes)


# --------------------------------------------------------------------------
# Têmpera Simulada
# --------------------------------------------------------------------------


def simulated_annealing(f, start=None, t0=5.0, t_min=0.01, cooling=0.9995,
                        step=30, rng=None):
    """Têmpera simulada com critério de Metropolis e resfriamento geométrico.

    Movimentos que pioram são aceitos com probabilidade exp(delta / T), o que
    permite escapar de ótimos locais enquanto a temperatura ainda é alta. O
    passo de mutação encolhe junto com a temperatura, de modo que o fim da
    execução funciona como um refinamento local.

    O melhor ponto já visto é guardado à parte: a têmpera vagueia e o estado
    final normalmente não é o melhor ponto da trajetória.
    """
    rng = rng or random
    current = normalize(start) if start is not None else random_point(rng)
    current_gain = f(current)
    best_point, best_gain = current, current_gain

    temperature = t0
    accepted = attempted = 0

    try:
        while temperature > t_min:
            scale = max(1, int(round(step * temperature / t0)))
            axis = rng.randrange(DIM)
            offset = int(round(rng.gauss(0, scale))) or rng.choice((-1, 1))
            candidate = perturb(current, axis, offset)
            gain = f(candidate)

            attempted += 1
            delta = gain - current_gain
            if delta >= 0 or rng.random() < math.exp(delta / temperature):
                current, current_gain = candidate, gain
                accepted += 1
                if gain > best_gain:
                    best_point, best_gain = candidate, gain

            temperature *= cooling
    except BudgetExhausted:
        pass

    rate = accepted / attempted if attempted else 0.0
    notes = f"{attempted} movimentos, {rate:.1%} aceitos, T final {temperature:.4f}"
    return Result("simulated_annealing", best_point, best_gain, f.evaluations, notes)


# --------------------------------------------------------------------------
# Algoritmo Genético
# --------------------------------------------------------------------------


def _tournament(population, fitness, k, rng):
    contenders = rng.sample(range(len(population)), k)
    winner = max(contenders, key=lambda i: fitness[i])
    return population[winner]


def _cyclic_crossover(parent_a, parent_b, rng):
    """Cruzamento que respeita a circularidade dos ângulos.

    Metade dos genes vem de um cruzamento uniforme (troca direta) e a outra
    metade de uma mistura: o filho fica entre os pais andando pelo MENOR arco.
    Sem isso, a média aritmética de 350 e 10 daria 180 — o lado oposto do
    círculo — em vez de 0.
    """
    child = []
    for a, b in zip(parent_a, parent_b):
        if rng.random() < 0.5:
            child.append(a if rng.random() < 0.5 else b)
        else:
            t = rng.random()
            child.append(int(round(a + t * signed_delta(a, b))) % 360)
    return tuple(child)


def _mutate(individual, rate, scale, rng):
    genes = list(individual)
    for i in range(DIM):
        if rng.random() < rate:
            genes[i] = (genes[i] + int(round(rng.gauss(0, scale)))) % 360
    return tuple(genes)


def genetic_algorithm(f, population_size=60, generations=200, elite=4,
                      tournament_k=3, mutation_rate=0.25, mutation_scale=25,
                      rng=None, seeds=()):
    """Algoritmo genético com elitismo, torneio e mutação gaussiana cíclica.

    A escala da mutação decai ao longo das gerações: começa exploratória e
    termina refinando. O elitismo garante que o melhor indivíduo nunca se perca.
    """
    rng = rng or random

    population = [normalize(s) for s in seeds][:population_size]
    population += [random_point(rng) for _ in range(population_size - len(population))]

    best_point, best_gain = None, float("-inf")
    generation = 0

    try:
        for generation in range(generations):
            fitness = [f(ind) for ind in population]

            ranked = sorted(zip(fitness, population), key=lambda p: p[0], reverse=True)
            if ranked[0][0] > best_gain:
                best_gain, best_point = ranked[0]

            scale = max(1, mutation_scale * (1 - generation / generations))
            offspring = [point for _, point in ranked[:elite]]
            while len(offspring) < population_size:
                a = _tournament(population, fitness, tournament_k, rng)
                b = _tournament(population, fitness, tournament_k, rng)
                offspring.append(_mutate(_cyclic_crossover(a, b, rng), mutation_rate, scale, rng))
            population = offspring
    except BudgetExhausted:
        pass

    diversity = len(set(population))
    notes = f"{generation + 1} gerações, {diversity}/{len(population)} indivíduos distintos ao final"
    return Result("genetic_algorithm", best_point, best_gain, f.evaluations, notes)


# --------------------------------------------------------------------------
# Descida coordenada exaustiva
# --------------------------------------------------------------------------


def exhaustive_coordinate_descent(f, start, max_sweeps=10):
    """Varre um eixo inteiro por vez, fixando os outros cinco.

    Como cada eixo tem apenas 360 valores possíveis, uma varredura completa
    custa 6 x 360 = 2160 avaliações e devolve o ótimo exato de cada eixo dado o
    resto. Com memoização, as varreduras seguintes ficam bem mais baratas.
    Converge quando uma varredura inteira não melhora mais nada.
    """
    current = normalize(start)
    current_gain = f(current)
    sweeps = 0

    try:
        for sweeps in range(1, max_sweeps + 1):
            improved = False
            for axis in range(DIM):
                best_value, best_gain = current[axis], current_gain
                for value in range(360):
                    candidate = list(current)
                    candidate[axis] = value
                    gain = f(tuple(candidate))
                    if gain > best_gain:
                        best_value, best_gain = value, gain
                if best_value != current[axis]:
                    current = perturb(current, axis, best_value - current[axis])
                    current_gain = best_gain
                    improved = True
            if not improved:
                break
    except BudgetExhausted:
        pass

    return Result("coordinate_descent", current, current_gain, f.evaluations,
                  f"{sweeps} varreduras completas")
