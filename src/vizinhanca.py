import random
import pandas as pd

# ============================================================
# Módulo: vizinhanca.py
# Descrição: Alínea c) — Define a estrutura de vizinhança
#            de uma solução admissível.
#
# Definição de vizinhança:
#   Dado que uma solução é um conjunto de 4 playlists, uma
#   solução vizinha é obtida através de um de dois movimentos:
#
#   Movimento 1 — Substituição (swap com disponíveis):
#     Seleciona aleatoriamente uma playlist e uma música dessa
#     playlist, e substitui-a por uma música do pool de músicas
#     disponíveis (não usadas), garantindo que a nova solução
#     permanece admissível.
#
#   Movimento 2 — Troca inter-playlist (swap entre playlists):
#     Seleciona aleatoriamente duas playlists distintas e troca
#     uma música de cada uma, garantindo que ambas as playlists
#     permanecem admissíveis após a troca.
#
#   A vizinhança N(s) de uma solução s é o conjunto de todas
#   as soluções obtidas por aplicação de um destes movimentos.
# ============================================================

from src.funcao_objetivo import (
    verificar_pl1, verificar_pl2, verificar_pl3, verificar_pl4, verificar_duracao
)

# Mapeamento das funções de verificação por playlist
VERIFICADORES = {
    'PL1': verificar_pl1,
    'PL2': verificar_pl2,
    'PL3': verificar_pl3,
    'PL4': verificar_pl4,
}


def verificar_playlist(nome_pl, ids, df):
    """
    Verifica se uma playlist é admissível (duração + restrição específica).
    Devolve True se admissível, False caso contrário.
    """
    ok_dur, _ = verificar_duracao(ids, df)
    if not ok_dur:
        return False
    ok_pl, _ = VERIFICADORES[nome_pl](ids, df)
    return ok_pl


def movimento_substituicao(solucao, df, musicas_disponiveis, max_tentativas=200):
    """
    Movimento 1 — Substituição:
    Remove uma música de uma playlist e substitui-a por uma música
    do pool de disponíveis. A nova solução tem de ser admissível.

    Parâmetros:
        solucao            — dicionário com as 4 playlists
        df                 — DataFrame com todas as músicas
        musicas_disponiveis — lista de track_ids não usados
        max_tentativas     — número máximo de tentativas antes de desistir

    Devolve:
        nova_solucao       — nova solução (ou a original se não encontrar vizinho)
        novas_disponiveis  — pool actualizado
        descricao          — texto descrevendo o movimento realizado
        sucesso            — True se encontrou vizinho admissível
    """
    import copy

    for _ in range(max_tentativas):
        # Escolher uma playlist aleatória
        nome_pl = random.choice(['PL1', 'PL2', 'PL3', 'PL4'])
        playlist_atual = solucao[nome_pl].copy()

        if len(playlist_atual) == 0 or len(musicas_disponiveis) == 0:
            continue

        # Escolher uma música aleatória da playlist para remover
        musica_saiu = random.choice(playlist_atual)

        # Escolher uma música aleatória do pool de disponíveis para entrar
        musica_entrou = random.choice(musicas_disponiveis)

        # Construir a nova playlist
        nova_playlist = [m for m in playlist_atual if m != musica_saiu]
        nova_playlist.append(musica_entrou)

        # Verificar se a nova playlist é admissível
        if verificar_playlist(nome_pl, nova_playlist, df):
            # Aceitar o movimento
            nova_solucao = copy.deepcopy(solucao)
            nova_solucao[nome_pl] = nova_playlist

            novas_disponiveis = [m for m in musicas_disponiveis if m != musica_entrou]
            novas_disponiveis.append(musica_saiu)

            musica_saiu_nome   = df[df['track_id'] == musica_saiu]['track_name'].values[0]
            musica_entrou_nome = df[df['track_id'] == musica_entrou]['track_name'].values[0]
            descricao = (f"Substituição em {nome_pl}: "
                         f"'{musica_saiu_nome}' → '{musica_entrou_nome}'")

            return nova_solucao, novas_disponiveis, descricao, True

    return solucao, musicas_disponiveis, "Substituição: nenhum vizinho admissível encontrado", False


def movimento_troca_interplaylist(solucao, df, musicas_disponiveis, max_tentativas=200):
    """
    Movimento 2 — Troca inter-playlist:
    Troca uma música entre duas playlists distintas.
    Ambas as playlists têm de permanecer admissíveis após a troca.

    Nota: este movimento não altera o pool de disponíveis.

    Devolve:
        nova_solucao  — nova solução (ou a original se não encontrar vizinho)
        descricao     — texto descrevendo o movimento realizado
        sucesso       — True se encontrou vizinho admissível
    """
    import copy

    playlists = ['PL1', 'PL2', 'PL3', 'PL4']

    for _ in range(max_tentativas):
        # Escolher duas playlists distintas
        pl_a, pl_b = random.sample(playlists, 2)

        lista_a = solucao[pl_a].copy()
        lista_b = solucao[pl_b].copy()

        if len(lista_a) == 0 or len(lista_b) == 0:
            continue

        # Escolher uma música aleatória de cada playlist
        musica_a = random.choice(lista_a)
        musica_b = random.choice(lista_b)

        # Construir as novas playlists com as músicas trocadas
        nova_lista_a = [m for m in lista_a if m != musica_a] + [musica_b]
        nova_lista_b = [m for m in lista_b if m != musica_b] + [musica_a]

        # Verificar admissibilidade das duas playlists afectadas
        if verificar_playlist(pl_a, nova_lista_a, df) and verificar_playlist(pl_b, nova_lista_b, df):
            nova_solucao = copy.deepcopy(solucao)
            nova_solucao[pl_a] = nova_lista_a
            nova_solucao[pl_b] = nova_lista_b

            nome_a = df[df['track_id'] == musica_a]['track_name'].values[0]
            nome_b = df[df['track_id'] == musica_b]['track_name'].values[0]
            descricao = (f"Troca inter-playlist: "
                         f"'{nome_a}' ({pl_a}) ↔ '{nome_b}' ({pl_b})")

            return nova_solucao, musicas_disponiveis, descricao, True

    return solucao, musicas_disponiveis, "Troca inter-playlist: nenhum vizinho admissível encontrado", False


def gerar_vizinho(solucao, df, musicas_disponiveis):
    """
    Gera uma solução vizinha escolhendo aleatoriamente entre os
    dois movimentos disponíveis (substituição ou troca inter-playlist).

    Usada pelo Simulated Annealing em cada iteração.

    Devolve:
        nova_solucao       — solução vizinha gerada
        novas_disponiveis  — pool actualizado
        descricao          — descrição do movimento realizado
        sucesso            — True se encontrou vizinho admissível
    """
    # Escolher aleatoriamente qual o movimento a aplicar
    tipo_movimento = random.choice(['substituicao', 'troca'])

    if tipo_movimento == 'substituicao':
        return movimento_substituicao(solucao, df, musicas_disponiveis)
    else:
        nova_sol, novas_disp, desc, ok = movimento_troca_interplaylist(
            solucao, df, musicas_disponiveis
        )
        return nova_sol, novas_disp, desc, ok

