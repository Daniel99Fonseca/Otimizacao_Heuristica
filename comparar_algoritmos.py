import pandas as pd
import random
random.seed(42)

from src.carrega_dados import carregar_dataset
from src.heuristica_construtiva import heuristica_construtiva
from src.funcao_objetivo import calcular_popularidade, verificar_admissibilidade
from src.simulated_annealing import simulated_annealing, M_ENUNCIADO, M_ALTERNATIVO
from src.algoritmo_genetico import algoritmo_genetico

# ============================================================
# comparar_algoritmos.py — Alínea f)
# Corre SA e AG a partir da mesma semente e compara resultados.
# Executar com: python comparar_algoritmos.py
# ============================================================

print("A carregar dataset...")
df = carregar_dataset()

# ------------------------------------------------------------------
# Solução inicial (heurística construtiva) — ponto de partida comum
# ------------------------------------------------------------------
random.seed(42)
sol_inicial, disponiveis = heuristica_construtiva(df)
pop_inicial = calcular_popularidade(sol_inicial, df)
print(f"Heurística construtiva — Popularidade: {pop_inicial}")

# ------------------------------------------------------------------
# SA — Esquema do enunciado
# ------------------------------------------------------------------
print("\n" + "=" * 55)
print("SA — Esquema do enunciado (M=[10,10,10,5,5], 40 iter.)")
print("=" * 55)
random.seed(42)
sol_sa_enun, pop_sa_enun = simulated_annealing(
    sol_inicial, df, disponiveis.copy(),
    "output/output_sa_enunciado.csv",
    esquema_M=M_ENUNCIADO
)

# ------------------------------------------------------------------
# SA — Esquema alternativo
# ------------------------------------------------------------------
print("\n" + "=" * 55)
print("SA — Esquema alternativo (380 iter.)")
print("=" * 55)
random.seed(42)
sol_sa_alt, pop_sa_alt = simulated_annealing(
    sol_inicial, df, disponiveis.copy(),
    "output/output_sa_alternativo.csv",
    esquema_M=M_ALTERNATIVO
)

# ------------------------------------------------------------------
# AG
# ------------------------------------------------------------------
print("\n" + "=" * 55)
print("AG — 20 cromossomas × 50 gerações")
print("=" * 55)
random.seed(42)
sol_ag, pop_ag = algoritmo_genetico(
    df,
    n_cromossomas  = 20,
    n_geracoes     = 50,
    prob_mutacao   = 0.1,
    caminho_output = "output/output_ag.csv",
    k_torneio      = 5
)

# ------------------------------------------------------------------
# Tabela comparativa final
# ------------------------------------------------------------------
print("\n" + "=" * 55)
print("COMPARAÇÃO FINAL — ALÍNEA f)")
print("=" * 55)

resultados = {
    'Heurística construtiva' : (sol_inicial,  pop_inicial),
    'SA — enunciado'         : (sol_sa_enun,  pop_sa_enun),
    'SA — alternativo'       : (sol_sa_alt,   pop_sa_alt),
    'AG'                     : (sol_ag,        pop_ag),
}

print(f"{'Algoritmo':<30} {'Popularidade':>14} {'Melhoria':>10} {'Admissível':>12}")
print("-" * 70)
for nome, (sol, pop) in resultados.items():
    adm, _ = verificar_admissibilidade(sol, df)
    melhoria = pop - pop_inicial
    sinal = f"+{melhoria}" if melhoria >= 0 else str(melhoria)
    print(f"{nome:<30} {pop:>14} {sinal:>10} {'✅' if adm else '❌':>12}")

print()

# Detalhe por playlist de cada solução
print("=" * 55)
print("DETALHE POR PLAYLIST")
print("=" * 55)
for nome, (sol, pop) in resultados.items():
    print(f"\n{nome}:")
    for pl, ids in sol.items():
        musicas     = df[df['track_id'].isin(ids)]
        dur_min     = musicas['duration_ms'].sum() / 60000
        pop_pl      = musicas['popularity'].sum()
        print(f"  {pl}: {len(ids):2d} músicas | {dur_min:.2f} min | pop = {pop_pl}")

# ------------------------------------------------------------------
# Guardar tabela de comparação em CSV
# ------------------------------------------------------------------
linhas = []
for nome, (sol, pop) in resultados.items():
    adm, _ = verificar_admissibilidade(sol, df)
    for pl, ids in sol.items():
        musicas = df[df['track_id'].isin(ids)]
        linhas.append({
            'algoritmo'   : nome,
            'playlist'    : pl,
            'n_musicas'   : len(ids),
            'duracao_min' : round(musicas['duration_ms'].sum() / 60000, 2),
            'popularidade': int(musicas['popularity'].sum()),
            'pop_total'   : pop,
            'admissivel'  : adm,
        })

df_comp = pd.DataFrame(linhas)
df_comp.to_csv("output/comparacao_algoritmos.csv", index=False)
print("\nTabela de comparação guardada em: output/comparacao_algoritmos.csv")
