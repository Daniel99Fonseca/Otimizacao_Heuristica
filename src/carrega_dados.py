import pandas as pd
import os

PASTA_SRC = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROJETO = os.path.dirname(PASTA_SRC)
CAMINHO_DATASET = os.path.join(RAIZ_PROJETO, 'data', 'dataset_playlist.csv')

# Duração mínima e máxima de cada playlist em milissegundos
DURACAO_MIN_MS = 32 * 60 * 1000   # 32 minutos
DURACAO_MAX_MS = 35 * 60 * 1000   # 35 minutos


def carregar_dataset():
    df = pd.read_csv(CAMINHO_DATASET)

    # Passo 1: remover duração zero
    df = df[df['duration_ms'] > 0]

    # Passo 2: remover nulos
    df = df.dropna()

    # Passo 3: identificar track_ids com pelo menos uma ocorrência acústica
    # Fazemos isto ANTES de deduplicar para não perder essa informação
    ids_acusticos = set(df[df['track_genre'] == 'acoustic']['track_id'].unique())
    df['is_acoustic'] = df['track_id'].isin(ids_acusticos)

    # Passo 4: deduplicar por track_id
    # Ordenamos por popularidade descendente e ficamos com a 1ª ocorrência
    # (a de maior popularidade). A coluna is_acoustic já está correcta
    # para todos os track_ids porque foi calculada antes da deduplicação.
    df = df.sort_values('popularity', ascending=False)
    df = df.drop_duplicates(subset='track_id', keep='first')

    # Passo 5: deduplicar por (artists, track_name)
    # Músicas com o mesmo nome e artista mas track_id diferente
    df = df.drop_duplicates(subset=['artists', 'track_name'], keep='first')

    # Géneros não adequados para rádio
    GENEROS_EXCLUIR = ['sleep', 'ambient', 'study']

    # Aplicar após deduplicação
    df = df[~df['track_genre'].isin(GENEROS_EXCLUIR)]

    # Filtro de features para apanhar musicas que escapam ao género
    df = df[
        ~(
            (df['energy']       < 0.05) &
            (df['danceability'] < 0.05) &
            (df['tempo']        < 10)
        )
    ]

    # Resetar o índice para ficar limpo
    df = df.reset_index(drop=True)

    return df


def filtrar_pl1(df):
    return df[
        (df['instrumentalness'] >= 0.66)
    ].copy()
    

def filtrar_pl2(df):
    return df[df['tempo'] >= 120].copy()


def filtrar_pl3_acusticas(df):
    return df[df['is_acoustic'] == True].copy()


def filtrar_pl3_aovivo(df):
    return df[df['liveness'] > 0.8].copy()


def filtrar_pl4(df):
    return df.sort_values('valence', ascending=False)
