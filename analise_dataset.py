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