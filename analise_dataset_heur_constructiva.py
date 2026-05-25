# ============================================================
# analise_dataset.py — Exploração inicial do dataset
# Executar com: python analise_dataset.py
# ============================================================

import pandas as pd
import os

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
# Análise por restrição de cada playlist
# -------------------------------------------------------

print("=" * 60)
print("CANDIDATAS POR PLAYLIST")
print("=" * 60)

# PL1: instrumentalness >= 0.66
pl1_cands = df[df['instrumentalness'] >= 0.66]
print(f"PL1 candidatas (instrumentalness >= 0.66): {len(pl1_cands)}")

# PL2: tempo >= 120 BPM
pl2_cands = df[df['tempo'] >= 120]
print(f"PL2 candidatas (tempo >= 120 BPM): {len(pl2_cands)}")

# PL3 acústicas (a definir threshold após análise)
print(f"\nPL3 — distribuição do acousticness:")
print(df['acousticness'].describe())

# PL3 ao vivo: liveness > 0.8
pl3_aovivo = df[df['liveness'] > 0.8]
print(f"\nPL3 candidatas ao vivo (liveness > 0.8): {len(pl3_aovivo)}")

# PL4: valence — qualquer música serve, mas queremos perceber a distribuição
print(f"\nPL4 — distribuição do valence:")
print(df['valence'].describe())

# -------------------------------------------------------
# Análise da duração
# -------------------------------------------------------

print("=" * 60)
print("ANÁLISE DA DURAÇÃO")
print("=" * 60)
print(f"Duração média (min): {df['duration_ms'].mean() / 60000:.2f}")
print(f"Duração mínima (min): {df['duration_ms'].min() / 60000:.2f}")
print(f"Duração máxima (min): {df['duration_ms'].max() / 60000:.2f}")
print()
print("Nota: cada playlist deve ter entre 32 e 35 minutos")
print(f"  → Estimativa de músicas por playlist: {32*60000 // int(df['duration_ms'].mean())}"
      f" a {35*60000 // int(df['duration_ms'].mean())}")


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
              f"dur={row['duration_ms']/60000:.2f} min")

print(f"\n{'='*60}")
print(f"Popularidade total: {pop_total}")

adm, msgs = verificar_admissibilidade(solucao, df)
print(f"\nVerificação de admissibilidade:")
for msg in msgs:
    print(f"  {msg}")