import pandas as pd

# Uma solução é representada como um dicionário de listas de track_ids:
#
# solucao = {
#     'PL1': ['id1', 'id2', ...],
#     'PL2': ['id3', 'id4', ...],
#     'PL3': ['id5', 'id6', ...],
#     'PL4': ['id7', 'id8', ...],
# }

DURACAO_MIN_MS = 32 * 60 * 1000   # 32 minutos em ms
DURACAO_MAX_MS = 35 * 60 * 1000   # 35 minutos em ms

# PL3: duração mínima de músicas acústicas
DURACAO_MIN_ACUSTICAS_MS = 15 * 60 * 1000  # 15 minutos em ms

# PL3: número mínimo de músicas ao vivo
MIN_MUSICAS_AOVIVO = 4

# PL4: valence total mínimo
VALENCE_TOTAL_MIN = 7.0


def calcular_popularidade(solucao, df):
    total = 0
    for playlist, ids in solucao.items():
        musicas = df[df['track_id'].isin(ids)]
        total += musicas['popularity'].sum()
    return int(total)


# ------------------------------------------------------------------
# Funções de verificação individuais por playlist
# ------------------------------------------------------------------

def verificar_duracao(ids, df):
    musicas = df[df['track_id'].isin(ids)]
    duracao_total_ms = musicas['duration_ms'].sum()
    duracao_min = duracao_total_ms / 60000
    ok = DURACAO_MIN_MS <= duracao_total_ms <= DURACAO_MAX_MS
    return ok, round(duracao_min, 2)


def verificar_pl1(ids, df):
    musicas = df[df['track_id'].isin(ids)]
    violacoes = musicas[musicas['instrumentalness'] < 0.66]
    if len(violacoes) > 0:
        return False, f"PL1: {len(violacoes)} músicas com instrumentalness < 0.66"
    return True, "PL1: OK"


def verificar_pl2(ids, df):
    musicas = df[df['track_id'].isin(ids)]

    # Verificação por música: tempo >= 120
    violacoes_tempo = musicas[musicas['tempo'] < 120]
    if len(violacoes_tempo) > 0:
        return False, f"PL2: {len(violacoes_tempo)} músicas com tempo < 120 BPM"

    # Verificação da playlist: danceability média >= 0.5
    dance_media = musicas['danceability'].mean()
    if dance_media < 0.5:
        return False, f"PL2: danceability média = {dance_media:.3f} < 0.5"

    return True, f"PL2: OK (danceability média = {dance_media:.3f})"


def verificar_pl3(ids, df):
    musicas = df[df['track_id'].isin(ids)]

    # Verificação das músicas acústicas (>= 15 min)
    acusticas = musicas[musicas['is_acoustic'] == True]
    duracao_acusticas_ms = acusticas['duration_ms'].sum()
    duracao_acusticas_min = duracao_acusticas_ms / 60000
    if duracao_acusticas_ms < DURACAO_MIN_ACUSTICAS_MS:
        return False, f"PL3: apenas {duracao_acusticas_min:.2f} min acústicos (mínimo 15 min)"

    # Verificação das músicas ao vivo (>= 4)
    aovivo = musicas[musicas['liveness'] > 0.8]
    if len(aovivo) < MIN_MUSICAS_AOVIVO:
        return False, f"PL3: apenas {len(aovivo)} músicas ao vivo (mínimo {MIN_MUSICAS_AOVIVO})"

    return True, f"PL3: OK ({duracao_acusticas_min:.2f} min acústicos, {len(aovivo)} ao vivo)"


def verificar_pl4(ids, df):
    musicas = df[df['track_id'].isin(ids)]
    valence_total = musicas['valence'].sum()
    if valence_total < VALENCE_TOTAL_MIN:
        return False, f"PL4: valence total = {valence_total:.3f} < {VALENCE_TOTAL_MIN}"
    return True, f"PL4: OK (valence total = {valence_total:.3f})"


# ------------------------------------------------------------------
# Verificação geral da solução completa
# ------------------------------------------------------------------

def verificar_admissibilidade(solucao, df):
    mensagens = []
    admissivel = True

    # Verificar se há músicas repetidas entre playlists
    todas_ids = []
    for ids in solucao.values():
        todas_ids.extend(ids)
    if len(todas_ids) != len(set(todas_ids)):
        admissivel = False
        mensagens.append("ERRO: existem músicas repetidas entre playlists!")

    # Verificar cada playlist
    verificacoes = {
        'PL1': verificar_pl1,
        'PL2': verificar_pl2,
        'PL3': verificar_pl3,
        'PL4': verificar_pl4,
    }

    for nome_pl, ids in solucao.items():
        # Duração
        ok_dur, dur_min = verificar_duracao(ids, df)
        if not ok_dur:
            admissivel = False
            mensagens.append(f"{nome_pl}: duração = {dur_min} min (deve estar entre 32 e 35 min)")
        else:
            mensagens.append(f"{nome_pl}: duração = {dur_min} min ✅")

        # Restrição específica da playlist
        ok_pl, msg_pl = verificacoes[nome_pl](ids, df)
        if not ok_pl:
            admissivel = False
        mensagens.append(msg_pl)

    return admissivel, mensagens

def guardar_melhor_solucao(solucao, df, caminho_output):
    linhas = []
    for pl, ids in solucao.items():
        musicas = df[df['track_id'].isin(ids)]
        for _, row in musicas.iterrows():
            linhas.append({
                'playlist'         : pl,
                'track_name'       : row['track_name'],
                'artists'          : row['artists'],
                'popularity'       : row['popularity'],
                'duracao_min'      : round(row['duration_ms'] / 60000, 2),
                'instrumentalness' : round(row['instrumentalness'], 3),
                'tempo'            : round(row['tempo'], 1),
                'danceability'     : round(row['danceability'], 3),
                'liveness'         : round(row['liveness'], 3),
                'valence'          : round(row['valence'], 3),
                'is_acoustic'      : row['is_acoustic'],
            })
    pd.DataFrame(linhas).to_csv(caminho_output, index=False)
    print(f"Melhor solução guardada em: {caminho_output}")