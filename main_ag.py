import random
import time 
import numpy as np

random.seed(42)
np.random.seed(42)

from src.carrega_dados import carregar_dataset
from src.algoritmo_genetico import algoritmo_genetico
from src.funcao_objetivo import calcular_popularidade, verificar_admissibilidade, guardar_melhor_solucao
from src.heuristica_construtiva import heuristica_construtiva

# ============================================================
# main_ag.py — Script principal do Algoritmo Genético
# Executar com: python main_ag.py
# ============================================================

# Parâmetros do AG
N_CROMOSSOMAS = 20
N_GERACOES    = 30
PROB_MUTACAO  = 0.1
K_TORNEIO     = 5

# --- 1. Carregar dados ---
print("A carregar dataset...")
df = carregar_dataset()
print(f"Dataset: {len(df)} músicas\n")

# Solução da heurística (para comparação)
sol_heuristica, _ = heuristica_construtiva(df)
pop_heuristica    = calcular_popularidade(sol_heuristica, df)
print(f"Solução heurística construtiva — Popularidade: {pop_heuristica}\n")

# --- 2. Correr o Algoritmo Genético ---
print("=" * 60)
print(f"ALGORITMO GENÉTICO")
print(f"População: {N_CROMOSSOMAS} | Gerações: {N_GERACOES} | "
      f"P(mutação): {PROB_MUTACAO} | Torneio k={K_TORNEIO}")
print("=" * 60 + "\n")

random.seed(42)             # ← adicionar
np.random.seed(42)          # ← adicionar

inicio_ag = time.time()
melhor_solucao, melhor_pop = algoritmo_genetico(
    df,
    n_cromossomas  = N_CROMOSSOMAS,
    n_geracoes     = N_GERACOES,
    prob_mutacao   = PROB_MUTACAO,
    caminho_output = "output/output_ag.csv",
    k_torneio      = K_TORNEIO
)
tempo_ag = time.time() - inicio_ag

# --- 3. Verificar admissibilidade ---
print("\n--- Verificação de Admissibilidade ---")
adm, msgs = verificar_admissibilidade(melhor_solucao, df)
for msg in msgs:
    print(f"  {msg}")

# --- 4. Detalhes da melhor solução ---
print("\n--- Composição da Melhor Solução AG ---")
for pl, ids in melhor_solucao.items():
    musicas     = df[df['track_id'].isin(ids)]
    duracao_min = musicas['duration_ms'].sum() / 60000
    pop_pl      = musicas['popularity'].sum()
    print(f"{pl}: {len(ids)} músicas | {duracao_min:.2f} min | popularidade = {pop_pl}")

# --- 5. Comparação final ---
print("\n" + "=" * 60)
print("RESUMO")
print("=" * 60)
print(f"Heurística construtiva: {pop_heuristica}")
print(f"Algoritmo Genético:     {melhor_pop}  ({melhor_pop - pop_heuristica:+d})")
print(f"Solução admissível:     {adm}")
print(f"AG — Tempo total: {tempo_ag:.2f} segundos")

guardar_melhor_solucao(melhor_solucao, df, "output/melhor_ag.csv")