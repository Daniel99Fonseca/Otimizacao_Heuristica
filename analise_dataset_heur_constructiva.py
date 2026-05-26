# ============================================================
# analise_dataset.py — Exploração inicial do dataset
# Executar com: python analise_dataset.py
# ============================================================

import pandas as pd
import os

from src.carrega_dados import carregar_dataset, filtrar_pl1, filtrar_pl2, filtrar_pl3_aovivo, filtrar_pl3_acusticas, filtrar_pl4

# Descobre a pasta onde o script atual está guardado
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Constrói o caminho para o dataset de forma robusta
CAMINHO_DATASET = os.path.join(BASE_DIR, 'data', 'dataset_playlist.csv')

df = pd.read_csv(CAMINHO_DATASET)

print("=" * 60)
print("ANÁLISE GERAL DO DATASET")
print("=" * 60)
print(f"Total de músicas (linhas): {len(df)}")
print(f"Colunas: {list(df.columns)}")
print()

print("--- Primeiras 5 linhas ---")
print(df.head())
print()

print("--- Tipos de dados ---")
print(df.dtypes)
print()

print("--- Estatísticas descritivas ---")
print(df.describe())
print()

print("--- Valores nulos ---")
print(df.isnull().sum())
print()

print("--- Duplicados (track_id) ---")
print(f"Track_ids duplicados: {df['track_id'].duplicated().sum()}")
print()


# -------------------------------------------------------
# Após limpeza 
# -------------------------------------------------------
print()
print("=" * 60)
print("APÓS LIMPEZA")
print("=" * 60)

df_2 = carregar_dataset()
print(f"Dataset após limpeza: {len(df_2)} músicas")
print(f"  - Candidatas PL1 (instrumentalness >= 0.66): {len(filtrar_pl1(df_2))}")
print(f"  - Candidatas PL2 (tempo >= 120 BPM): {len(filtrar_pl2(df_2))}")
print(f"  - Candidatas PL3 acústicas: {len(filtrar_pl3_acusticas(df_2))}")
print(f"  - Candidatas PL3 ao vivo: {len(filtrar_pl3_aovivo(df_2))}")
print(f"  - Candidatas PL4: {len(filtrar_pl4(df_2))}")

import random
random.seed(42)

from src.carrega_dados import carregar_dataset
from src.heuristica_construtiva import heuristica_construtiva
from src.funcao_objetivo import calcular_popularidade, verificar_admissibilidade

df = carregar_dataset()
solucao, disponiveis = heuristica_construtiva(df)
pop_total = calcular_popularidade(solucao, df)

print("=" * 60)
print("HEURÍSTICA CONSTRUTIVA — SOLUÇÃO INICIAL")
print("=" * 60)

for pl, ids in solucao.items():
    musicas = df[df['track_id'].isin(ids)]
    dur_min = musicas['duration_ms'].sum() / 60000
    pop_pl  = musicas['popularity'].sum()
    print(f"\n{pl}: {len(ids)} músicas | {dur_min:.2f} min | popularidade = {pop_pl}")
    for _, row in musicas.sort_values('popularity', ascending=False).iterrows():
        print(f"  {row['track_name']:<45} | pop={row['popularity']:3d} | "
              f"dur={row['duration_ms']/60000:.2f} min | valence={row['valence']:.3f}")

print(f"\n{'='*60}")
print(f"Popularidade total: {pop_total}")

adm, msgs = verificar_admissibilidade(solucao, df)
print(f"\nVerificação de admissibilidade:")
for msg in msgs:
    print(f"  {msg}")
