import random
import numpy as np
import pandas as pd

# ============================================================
# Módulo: algoritmo_genetico.py
# Descrição: Alínea e) — Algoritmo Genético.
# ============================================================


def gerar_populacao_inicial(df, n_cromossomas):
    """
    Gera a população inicial com n_cromossomas soluções admissíveis,
    usando a heurística construtiva com alguma aleatoriedade.
    """
    pass


def selecionar_pai(populacao):
    """
    Seleção por torneio: escolhe k cromossomas aleatoriamente
    e devolve o de maior aptidão (maior popularidade).
    """
    pass


def crossover(pai1, pai2, df):
    """
    Operador de crossover entre dois cromossomas.
    Gera dois filhos garantindo que as restrições são respeitadas.
    """
    pass


def mutacao(cromossoma, df, musicas_disponiveis, prob_mutacao):
    """
    Operador de mutação: com probabilidade prob_mutacao,
    substitui uma música aleatória de uma playlist por outra disponível.
    """
    pass


def algoritmo_genetico(df, n_cromossomas, n_geracoes, prob_mutacao, caminho_output):
    """
    Executa o Algoritmo Genético completo.
    Devolve a melhor solução encontrada e o seu valor de função objetivo.
    """
    pass
