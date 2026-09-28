#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface com o simulador de antena (OPServer.jar).

A função objetivo é o ganho retornado pelo servidor para uma configuração de
3 junções, cada uma com um par de ângulos (phi, theta) inteiros em [0, 360).

Duas características do simulador guiam o desenho desta classe:

1. Ele é determinístico: a mesma configuração sempre devolve o mesmo ganho.
   Logo todo ponto avaliado é memoizado, e o cache é gravado em disco para que
   execuções seguintes do agente não paguem de novo pelos mesmos pontos.
2. Ele é lento em relação à busca (~17 ms por requisição, servidor single-thread).
   O número de requisições é o recurso escasso, então `Antenna` contabiliza
   avaliações reais (cache miss) separadamente dos acertos de cache.

Os ângulos são cíclicos: o servidor trata 360 como 0, e rejeita negativos. Por
isso toda configuração é normalizada com % 360 antes de ser enviada, o que deixa
os algoritmos de busca livres para somar/subtrair sem se preocupar com limites.
"""

import http.client
import json
import os
import time

HOST = "localhost"
PORT = 8080
PATH = "/antenna/simulate?phi1=%d&theta1=%d&phi2=%d&theta2=%d&phi3=%d&theta3=%d"

DIM = 6
LABELS = ["phi1", "theta1", "phi2", "theta2", "phi3", "theta3"]

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache.json")


def normalize(angles):
    """Leva uma configuração qualquer para o representante canônico em [0, 360)."""
    return tuple(int(a) % 360 for a in angles)


class BudgetExhausted(Exception):
    """Sinaliza que o limite de avaliações reais foi atingido."""


class Antenna:
    """Função objetivo chamável: `gain = antenna([phi1, theta1, ...])`.

    `cache_file=None` mantém o cache apenas em memória, sem ler nem gravar em
    disco — é o que torna justa a comparação entre algoritmos, já que cada um
    precisa pagar pelas próprias avaliações.
    """

    def __init__(self, budget=None, cache_file=CACHE_FILE, verbose=True):
        self.budget = budget
        self.cache_file = cache_file
        self.verbose = verbose

        self.cache = {}
        self.evaluations = 0
        self.cache_hits = 0
        self.best_gain = float("-inf")
        self.best_point = None
        self.history = []

        self._conn = None
        self._unsaved = 0
        self._started = time.time()

        self._load_cache()

    # ---------------------------------------------------------------- cache

    def _load_cache(self):
        if self.cache_file is None or not os.path.exists(self.cache_file):
            return
        with open(self.cache_file, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self.cache = {tuple(int(v) for v in k.split(",")): g for k, g in raw.items()}
        if self.cache and self.verbose:
            print(f" - cache carregado: {len(self.cache)} pontos já conhecidos")
        for point, gain in self.cache.items():
            if gain > self.best_gain:
                self.best_gain, self.best_point = gain, point

    def save_cache(self):
        if self.cache_file is None:
            return
        raw = {",".join(str(v) for v in k): g for k, g in self.cache.items()}
        tmp = self.cache_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(raw, fh)
        os.replace(tmp, self.cache_file)
        self._unsaved = 0

    # ----------------------------------------------------------------- http

    def _request(self, point):
        for attempt in range(4):
            try:
                if self._conn is None:
                    self._conn = http.client.HTTPConnection(HOST, PORT, timeout=15)
                self._conn.request("GET", PATH % point)
                body = self._conn.getresponse().read().decode("utf-8", "replace")
            except (http.client.HTTPException, OSError):
                # Conexão keep-alive caída: descarta e tenta de novo.
                self._close()
                if attempt == 3:
                    raise
                time.sleep(0.2 * (attempt + 1))
                continue

            first_line = body.split("\n")[0].strip()
            try:
                return float(first_line)
            except ValueError:
                raise RuntimeError(f"resposta inesperada do servidor para {point}: {first_line!r}")
        raise RuntimeError("falha ao contatar o servidor")

    def _close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ------------------------------------------------------------ avaliação

    def __call__(self, angles):
        point = normalize(angles)

        if point in self.cache:
            self.cache_hits += 1
            return self.cache[point]

        if self.budget is not None and self.evaluations >= self.budget:
            raise BudgetExhausted(f"orçamento de {self.budget} avaliações esgotado")

        gain = self._request(point)
        self.cache[point] = gain
        self.evaluations += 1
        self._unsaved += 1

        if self._unsaved >= 200:
            self.save_cache()

        if gain > self.best_gain:
            self.best_gain, self.best_point = gain, point
            self.history.append((self.evaluations, gain, point))
            if self.verbose:
                print(f"   [{self.evaluations:6d}] novo melhor: {gain:.6f}  {self.describe(point)}")

        return gain

    def remaining(self):
        if self.budget is None:
            return float("inf")
        return max(0, self.budget - self.evaluations)

    # -------------------------------------------------------------- relatos

    @staticmethod
    def describe(point):
        return " ".join(f"{name}={value}" for name, value in zip(LABELS, point))

    def report(self):
        elapsed = time.time() - self._started
        total = self.evaluations + self.cache_hits
        print()
        print("=" * 70)
        print(f" melhor ganho : {self.best_gain:.10f}")
        print(f" configuração : {self.describe(self.best_point)}")
        print(f" avaliações   : {self.evaluations} reais + {self.cache_hits} de cache = {total}")
        print(f" tempo        : {elapsed:.1f}s")
        print("=" * 70)

    def close(self):
        self.save_cache()
        self._close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
