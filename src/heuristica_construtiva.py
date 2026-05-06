from src.carrega_dados import (
    filtrar_pl1, filtrar_pl2,
    filtrar_pl3_acusticas, filtrar_pl3_aovivo,
    filtrar_pl4,
    DURACAO_MIN_MS, DURACAO_MAX_MS
)

# ============================================================
# Módulo: heuristica_construtiva.py
# Descrição: Alínea b) — Heurística construtiva greedy para
#            gerar uma solução inicial admissível.
#
# Critério greedy: em cada playlist, as músicas são adicionadas
# por ordem decrescente de popularidade (maximizar o objetivo),
# respeitando sempre as restrições da playlist e os limites
# de duração (32-35 min). Músicas já usadas numa playlist não
# podem ser usadas noutra.
#
# Ordem de construção:
#   1. PL1 — mais restritiva em termos de instrumentalness
#   2. PL2 — restrita por tempo e danceability
#   3. PL3 — a mais complexa: requer músicas acústicas E ao vivo
#   4. PL4 — requer valence total elevado
# ============================================================

DURACAO_MIN_ACUSTICAS_MS = 15 * 60 * 1000  # 15 minutos em ms
MIN_MUSICAS_AOVIVO = 4


def construir_pl1(df, ids_usados):
    """
    Constrói a PL1: seleciona músicas com instrumentalness >= 0.66,
    ordenadas por popularidade descendente, até atingir 32-35 min.
    """
    candidatas = filtrar_pl1(df).sort_values('popularity', ascending=False)

    playlist = []
    duracao_atual = 0

    for _, row in candidatas.iterrows():
        if row['track_id'] in ids_usados:
            continue
        # Só adiciona se não ultrapassar os 35 minutos
        if duracao_atual + row['duration_ms'] <= DURACAO_MAX_MS:
            playlist.append(row['track_id'])
            duracao_atual += row['duration_ms']
            ids_usados.add(row['track_id'])
        # Para quando atingir os 32 minutos mínimos
        if duracao_atual >= DURACAO_MIN_MS:
            break

    return playlist


def construir_pl2(df, ids_usados):
    """
    Constrói a PL2: seleciona músicas com tempo >= 120 BPM,
    ordenadas por popularidade descendente, até atingir 32-35 min.
    A danceability média >= 0.5 é verificada a posteriori —
    dado que a média do pool PL2 é 0.562, é expectável que
    a restrição seja satisfeita naturalmente.
    """
    candidatas = filtrar_pl2(df).sort_values('popularity', ascending=False)

    playlist = []
    duracao_atual = 0

    for _, row in candidatas.iterrows():
        if row['track_id'] in ids_usados:
            continue
        if duracao_atual + row['duration_ms'] <= DURACAO_MAX_MS:
            playlist.append(row['track_id'])
            duracao_atual += row['duration_ms']
            ids_usados.add(row['track_id'])
        if duracao_atual >= DURACAO_MIN_MS:
            break

    return playlist


def construir_pl3(df, ids_usados):
    """
    Constrói a PL3: a mais complexa, com duas sub-restrições.

    Estratégia de construção (por fases):
      Fase 1 — Garantir >= 4 músicas ao vivo (liveness > 0.8),
               ordenadas por popularidade descendente.
      Fase 2 — Garantir >= 15 min de músicas acústicas (is_acoustic),
               ordenadas por popularidade descendente.
               Nota: uma música pode satisfazer ambas as fases
               simultaneamente (ao vivo E acústica).
      Fase 3 — Completar a duração até [32, 35] min com qualquer
               música por ordem de popularidade, sem restrições
               adicionais (exceto não ser já usada).
    """
    candidatas_aovivo   = filtrar_pl3_aovivo(df).sort_values('popularity', ascending=False)
    candidatas_acusticas = filtrar_pl3_acusticas(df).sort_values('popularity', ascending=False)

    playlist = []
    duracao_atual = 0

    # --- Fase 1: garantir >= 4 músicas ao vivo ---
    n_aovivo = 0
    for _, row in candidatas_aovivo.iterrows():
        if row['track_id'] in ids_usados:
            continue
        if duracao_atual + row['duration_ms'] <= DURACAO_MAX_MS:
            playlist.append(row['track_id'])
            duracao_atual += row['duration_ms']
            ids_usados.add(row['track_id'])
            n_aovivo += 1
        if n_aovivo >= MIN_MUSICAS_AOVIVO:
            break

    # --- Fase 2: garantir >= 15 min de músicas acústicas ---
    # Calculamos os minutos acústicos já garantidos na fase 1
    musicas_na_pl = df[df['track_id'].isin(playlist)]
    duracao_acusticas = musicas_na_pl[musicas_na_pl['is_acoustic'] == True]['duration_ms'].sum()

    for _, row in candidatas_acusticas.iterrows():
        if row['track_id'] in ids_usados:
            continue
        if duracao_acusticas >= DURACAO_MIN_ACUSTICAS_MS:
            break
        if duracao_atual + row['duration_ms'] <= DURACAO_MAX_MS:
            playlist.append(row['track_id'])
            duracao_atual += row['duration_ms']
            ids_usados.add(row['track_id'])
            duracao_acusticas += row['duration_ms']

    # --- Fase 3: completar a duração até 32 min com músicas populares ---
    todas_ordenadas = df.sort_values('popularity', ascending=False)
    for _, row in todas_ordenadas.iterrows():
        if duracao_atual >= DURACAO_MIN_MS:
            break
        if row['track_id'] in ids_usados:
            continue
        if duracao_atual + row['duration_ms'] <= DURACAO_MAX_MS:
            playlist.append(row['track_id'])
            duracao_atual += row['duration_ms']
            ids_usados.add(row['track_id'])

    return playlist


def construir_pl4(df, ids_usados):
    """
    Constrói a PL4: seleciona músicas com valence mais alto,
    até atingir 32-35 min. A restrição é valence total >= 7.0.
    Ordenar por valence descendente garante que atingimos
    facilmente esse total.
    """
    candidatas = filtrar_pl4(df)  # já ordenado por valence desc

    playlist = []
    duracao_atual = 0

    for _, row in candidatas.iterrows():
        if row['track_id'] in ids_usados:
            continue
        if duracao_atual + row['duration_ms'] <= DURACAO_MAX_MS:
            playlist.append(row['track_id'])
            duracao_atual += row['duration_ms']
            ids_usados.add(row['track_id'])
        if duracao_atual >= DURACAO_MIN_MS:
            break

    return playlist


def heuristica_construtiva(df):
    """
    Constrói uma solução admissível para as 4 playlists usando
    uma heurística greedy (critério: maximizar popularidade).

    Devolve:
        solucao           — dicionário {'PL1': [...], ..., 'PL4': [...]}
        musicas_disponiveis — lista de track_ids não usados em nenhuma playlist
                              (necessária para os algoritmos de pesquisa)
    """
    ids_usados = set()

    pl1 = construir_pl1(df, ids_usados)
    pl2 = construir_pl2(df, ids_usados)
    pl3 = construir_pl3(df, ids_usados)
    pl4 = construir_pl4(df, ids_usados)

    solucao = {
        'PL1': pl1,
        'PL2': pl2,
        'PL3': pl3,
        'PL4': pl4,
    }

    # Músicas que não foram usadas em nenhuma playlist
    musicas_disponiveis = list(df[~df['track_id'].isin(ids_usados)]['track_id'])

    return solucao, musicas_disponiveis
