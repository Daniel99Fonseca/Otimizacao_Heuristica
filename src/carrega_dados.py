import pandas as pd
import os

# ============================================================
# Módulo: carrega_dados.py
# Descrição: Leitura do dataset e pré-filtro das músicas
#            candidatas para cada playlist.
# ============================================================

# 1. Localização do ficheiro atual (C:\...\Otimizacao_Heuristica\src)
PASTA_SRC = os.path.dirname(os.path.abspath(__file__))

# 2. Sobe um nível para chegar à raiz do projeto (C:\...\Otimizacao_Heuristica)
RAIZ_PROJETO = os.path.dirname(PASTA_SRC)

# 3. Constrói o caminho para a pasta data que está na raiz
CAMINHO_DATASET = os.path.join(RAIZ_PROJETO, 'data', 'dataset_playlist.csv')

# Duração mínima e máxima de cada playlist (em milissegundos)
DURACAO_MIN_MS = 32 * 60 * 1000   # 32 minutos
DURACAO_MAX_MS = 35 * 60 * 1000   # 35 minutos


def carregar_dataset():
    """
    Lê o ficheiro CSV, aplica o pipeline de limpeza e devolve
    um DataFrame limpo e deduplicado com todas as músicas.

    Pipeline de limpeza (por ordem):
        1. Remover músicas com duration_ms == 0 (erros de dados)
        2. Remover linhas com valores nulos
        3. Antes de deduplicar, identificar quais os track_ids
           que têm pelo menos uma ocorrência com track_genre == 'acoustic'
           e guardar numa coluna auxiliar 'is_acoustic'
        4. Deduplicar por track_id (ficamos com a de maior popularidade)
        5. Deduplicar por (artists, track_name) (ficamos com a de maior popularidade)
    """
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
    # (ex: single vs álbum) — ficamos com a de maior popularidade
    df = df.drop_duplicates(subset=['artists', 'track_name'], keep='first')

    # Géneros não adequados para rádio convencional
    GENEROS_EXCLUIR = ['sleep', 'ambient', 'study']

    # Aplicar após deduplicação
    df = df[~df['track_genre'].isin(GENEROS_EXCLUIR)]

    # Filtro de features para apanhar casos que escapam ao género
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
    """
    PL1: instrumentalness >= 0.66
    Devolve apenas as músicas candidatas à PL1.
    """
    return df[
        (df['instrumentalness'] >= 0.66)
    ].copy()
    

def filtrar_pl2(df):
    """
    PL2: tempo >= 120 BPM (restrição individual por música)
    Nota: a condição de danceability média >= 0.5 é uma restrição
    ao nível da solução (playlist completa), não por música individual.
    Por isso não filtramos por danceability aqui.
    """
    return df[df['tempo'] >= 120].copy()


def filtrar_pl3_acusticas(df):
    """
    PL3 — componente acústica: músicas com track_genre == 'acoustic'
    (identificadas pela coluna is_acoustic criada no carregamento)
    """
    return df[df['is_acoustic'] == True].copy()


def filtrar_pl3_aovivo(df):
    """
    PL3 — componente ao vivo: liveness > 0.8
    Conforme o enunciado: "um valor superior a 0,8 indica uma forte
    probabilidade de a faixa ter sido gravada ao vivo"
    """
    return df[df['liveness'] > 0.8].copy()


def filtrar_pl4(df):
    """
    PL4: não existe filtro individual por música.
    A restrição (valence total >= 7.0) é verificada ao nível da
    playlist completa. No entanto, para a heurística construtiva
    ser eficaz, convém selecionar músicas com valence elevado.
    Por isso devolvemos o dataset completo ordenado por valence
    descendente para facilitar a construção.
    """
    return df.sort_values('valence', ascending=False)
