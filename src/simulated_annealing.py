import random
import numpy as np
import pandas as pd

# ============================================================
# Módulo: simulated_annealing.py
# Descrição: Alínea d) — Algoritmo Simulated Annealing.
#
# Esquema de arrefecimento definido no enunciado:
#   M = [10, 10, 10, 5, 5]  → total de 40 iterações
#   t_{i+1} = 0.3 * t_i
#   t_0 = 0.01 * f(solucao_inicial)
#
# Para MAXIMIZAÇÃO da popularidade:
#   delta = f(vizinho) - f(atual)
#   - Se delta > 0: aceitar sempre (melhoria)
#   - Se delta <= 0: aceitar com probabilidade exp(delta / T)
#     (delta é negativo, logo a probabilidade está entre 0 e 1)
# ============================================================

from src.vizinhanca import gerar_vizinho
from src.funcao_objetivo import calcular_popularidade

# Esquema de arrefecimento do enunciado
M_ENUNCIADO       = [10, 10, 10, 5, 5]
TAXA_ARREFECIMENTO = 0.3

# Esquema alternativo (mais iterações para explorar melhor)
M_ALTERNATIVO = [30, 30, 40, 50, 50, 40, 30]

def simulated_annealing(solucao_inicial, df, musicas_disponiveis,
                        caminho_output, esquema_M=None):
    """
    Executa o algoritmo Simulated Annealing para maximizar a
    popularidade total das playlists.

    Parâmetros:
        solucao_inicial    — dicionário com as 4 playlists (da heurística construtiva)
        df                 — DataFrame com todas as músicas
        musicas_disponiveis — lista de track_ids não usados na solução inicial
        caminho_output     — caminho para o ficheiro CSV de log
        esquema_M          — lista com nº de iterações por temperatura
                             (usa M_ENUNCIADO se None)

    Devolve:
        melhor_solucao — melhor solução encontrada
        melhor_pop     — popularidade total da melhor solução
    """
    if esquema_M is None:
        esquema_M = M_ENUNCIADO

    # --- Inicialização ---
    f_inicial   = calcular_popularidade(solucao_inicial, df)
    temperatura = 0.01 * f_inicial          # t_0 = 0.01 * f(sol_inicial)

    sol_atual   = solucao_inicial
    disp_atual  = musicas_disponiveis
    f_atual     = f_inicial

    melhor_solucao = sol_atual
    melhor_pop     = f_atual

    print(f"SA iniciado | Popularidade inicial: {f_inicial} | t_0 = {temperatura:.4f}")
    print(f"Esquema M: {esquema_M} | Total iterações: {sum(esquema_M)}")
    print(f"Taxa de arrefecimento: {TAXA_ARREFECIMENTO}\n")

    # Log de todas as iterações (para o ficheiro de output exigido pelo enunciado)
    log = []
    iteracao_global = 0

    # --- Ciclo principal: para cada temperatura ---
    for k, n_iter in enumerate(esquema_M):

        print(f"Temperatura {k+1}/{len(esquema_M)}: t = {temperatura:.6f} | Iterações: {n_iter}")

        for _ in range(n_iter):
            iteracao_global += 1

            # Gerar um vizinho aleatório admissível
            nova_sol, novo_disp, descricao, sucesso = gerar_vizinho(
                sol_atual, df, disp_atual
            )

            if not sucesso:
                # Não foi encontrado vizinho — registar e continuar
                log.append({
                    'iteracao'              : iteracao_global,
                    'temperatura'           : round(temperatura, 6),
                    'f_atual'               : f_atual,
                    'f_vizinho'             : '-',
                    'delta'                 : '-',
                    'prob_aceitacao'        : '-',
                    'aceite'                : 'Sem vizinho',
                    'movimento'             : descricao,
                    'melhor_ate_agora'      : melhor_pop,
                })
                continue

            f_vizinho = calcular_popularidade(nova_sol, df)
            delta     = f_atual - f_vizinho   # negativo = melhoria

            # --- Critério de aceitação ---
            if delta < 0:
                # Melhoria: aceitar sempre
                aceite         = True
                prob_aceitacao = 1.0
            else:
                # Piora: aceitar com probabilidade exp(delta / T)
                prob_aceitacao = np.exp(-delta / temperatura)
                aceite         = random.random() < prob_aceitacao

            # Registar no log
            log.append({
                'iteracao'         : iteracao_global,
                'temperatura'      : round(temperatura, 6),
                'f_atual'          : f_atual,
                'f_vizinho'        : f_vizinho,
                'delta'            : delta,
                'prob_aceitacao'   : round(prob_aceitacao, 6),
                'aceite'           : 'Sim' if aceite else 'Não',
                'movimento'        : descricao,
                'melhor_ate_agora' : melhor_pop,
            })

            # Actualizar solução actual se aceite
            if aceite:
                sol_atual  = nova_sol
                disp_atual = novo_disp
                f_atual    = f_vizinho

                # Actualizar melhor solução global
                if f_atual > melhor_pop:
                    melhor_solucao = sol_atual
                    melhor_pop     = f_atual
                    print(f"  ✅ Nova melhor solução: {melhor_pop} (iteração {iteracao_global})")

        # Arrefecer a temperatura
        temperatura = TAXA_ARREFECIMENTO * temperatura

    # --- Guardar log no ficheiro de output ---
    df_log = pd.DataFrame(log)
    df_log.to_csv(caminho_output, index=False)
    print(f"\nLog guardado em: {caminho_output}")
    print(f"Melhor solução SA — Popularidade: {melhor_pop}")

    return melhor_solucao, melhor_pop

