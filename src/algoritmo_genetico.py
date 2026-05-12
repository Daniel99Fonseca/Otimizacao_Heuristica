import random
import copy
import pandas as pd

# ============================================================
# Módulo: algoritmo_genetico.py
# Descrição: Alínea e) — Algoritmo Genético.
#
# Codificação: cada cromossoma é um dicionário com as 4 playlists
#   cromossoma = {'PL1': [ids], 'PL2': [ids], 'PL3': [ids], 'PL4': [ids]}
#
# População inicial: N soluções geradas pela heurística construtiva
#   com candidatos baralháveis aleatoriamente (diversidade).
#
# Selecção: torneio de tamanho k — escolhem-se k cromossomas
#   aleatoriamente e o vencedor é o de maior popularidade.
#
# Crossover: crossover ao nível das playlists —
#   o filho1 herda PL1+PL2 do pai1 e PL3+PL4 do pai2,
#   o filho2 herda PL1+PL2 do pai2 e PL3+PL4 do pai1.
#   Um mecanismo de reparação resolve conflitos (músicas repetidas).
#
# Mutação: com probabilidade prob_mutacao, aplica um movimento
#   de substituição (do módulo vizinhança) ao cromossoma.
#
# Cromossomas não admissíveis: são reparados pelo mecanismo
#   de reparação de conflitos após o crossover.
#
# Substituição da população: modelo geracional — a nova geração
#   substitui completamente a anterior, com elitismo (o melhor
#   cromossoma da geração anterior é sempre preservado).
#
# Critério de paragem: número máximo de gerações atingido.
# ============================================================

from src.heuristica_construtiva import (
    construir_pl1, construir_pl2, construir_pl3, construir_pl4,
    DURACAO_MIN_ACUSTICAS_MS, MIN_MUSICAS_AOVIVO
)
from src.funcao_objetivo import calcular_popularidade
from src.vizinhanca import movimento_substituicao
from src.carrega_dados import (
    filtrar_pl1, filtrar_pl2, filtrar_pl3_acusticas,
    filtrar_pl3_aovivo, filtrar_pl4,
    DURACAO_MIN_MS, DURACAO_MAX_MS
)


# ------------------------------------------------------------------
# População inicial
# ------------------------------------------------------------------

def gerar_cromossoma(df):
    """
    Gera um cromossoma admissível usando a heurística construtiva,
    mas com os candidatos de cada playlist baralháveis aleatoriamente
    para introduzir diversidade na população inicial.
    Devolve (cromossoma, musicas_disponiveis).
    """
    ids_usados = set()

    def candidatos_baralhados(df_filtrado, top_n=500):
        # Mistura as top 500 músicas aleatoriamente, seguidas das restantes
        top     = df_filtrado.nlargest(top_n, 'popularity')
        resto   = df_filtrado[~df_filtrado.index.isin(top.index)]
        top_bar = top.sample(frac=1).reset_index(drop=True)
        return pd.concat([top_bar, resto], ignore_index=True)

    pl1 = construir_pl1(candidatos_baralhados(filtrar_pl1(df)), ids_usados)
    pl2 = construir_pl2(candidatos_baralhados(filtrar_pl2(df)), ids_usados)
    pl3 = construir_pl3_ag(df, ids_usados)
    pl4 = construir_pl4(candidatos_baralhados(filtrar_pl4(df)), ids_usados)

    cromossoma  = {'PL1': pl1, 'PL2': pl2, 'PL3': pl3, 'PL4': pl4}
    disponiveis = list(df[~df['track_id'].isin(ids_usados)]['track_id'])

    return cromossoma, disponiveis


def construir_pl3_ag(df, ids_usados):
    """
    Versão da construção PL3 para o AG (recebe df completo).
    Lógica igual à do módulo heuristica_construtiva.
    """
    candidatas_aovivo    = filtrar_pl3_aovivo(df).sort_values('popularity', ascending=False)
    candidatas_acusticas = filtrar_pl3_acusticas(df).sort_values('popularity', ascending=False)

    playlist      = []
    duracao_atual = 0
    n_aovivo      = 0

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

    musicas_na_pl     = df[df['track_id'].isin(playlist)]
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


def gerar_populacao_inicial(df, n_cromossomas):
    """
    Gera a população inicial com n_cromossomas soluções admissíveis.
    Devolve uma lista de tuplos (cromossoma, musicas_disponiveis).
    """
    populacao = []
    for _ in range(n_cromossomas):
        cromossoma, disponiveis = gerar_cromossoma(df)
        populacao.append((cromossoma, disponiveis))
    return populacao


# ------------------------------------------------------------------
# Selecção por torneio
# ------------------------------------------------------------------

def selecionar_pai(populacao, df, k=5):
    """
    Selecção por torneio: escolhe k cromossomas aleatoriamente
    e devolve o índice do de maior popularidade (aptidão).
    """
    candidatos = random.sample(range(len(populacao)), k)
    melhor_idx = max(candidatos, key=lambda i: calcular_popularidade(populacao[i][0], df))
    return melhor_idx


# ------------------------------------------------------------------
# Crossover e reparação
# ------------------------------------------------------------------

def reparar_conflitos(cromossoma, df):
    """
    Após o crossover, uma música pode aparecer em mais do que uma
    playlist. Este mecanismo de reparação:
      1. Detecta músicas duplicadas entre playlists
      2. Remove os duplicados (mantém na primeira playlist onde aparece)
      3. Tenta repor músicas em playlists que ficaram curtas (< 32 min)
    Devolve o cromossoma reparado e o pool de disponíveis actualizado.
    """
    # 1. Detectar e resolver duplicados
    contagem = {}
    for pl, ids in cromossoma.items():
        for mid in ids:
            contagem[mid] = contagem.get(mid, []) + [pl]

    duplicadas = {mid: pls for mid, pls in contagem.items() if len(pls) > 1}
    for mid, pls in duplicadas.items():
        for pl_remover in pls[1:]:
            cromossoma[pl_remover] = [m for m in cromossoma[pl_remover] if m != mid]

    # 2. Reconstruir pool de disponíveis
    ids_usados  = set(mid for ids in cromossoma.values() for mid in ids)
    disponiveis = list(df[~df['track_id'].isin(ids_usados)]['track_id'])

    # 3. Repor músicas em playlists que ficaram curtas
    candidatas_disp = df[df['track_id'].isin(disponiveis)].sort_values('popularity', ascending=False)

    for pl in ['PL1', 'PL2', 'PL3', 'PL4']:
        ids_pl     = cromossoma[pl]
        duracao_ms = df[df['track_id'].isin(ids_pl)]['duration_ms'].sum()

        for _, row in candidatas_disp.iterrows():
            if duracao_ms >= DURACAO_MIN_MS:
                break
            if row['track_id'] in ids_usados:
                continue
            if duracao_ms + row['duration_ms'] <= DURACAO_MAX_MS:
                ids_pl.append(row['track_id'])
                ids_usados.add(row['track_id'])
                duracao_ms += row['duration_ms']

        cromossoma[pl] = ids_pl

    # 4. Pool final actualizado
    ids_usados  = set(mid for ids in cromossoma.values() for mid in ids)
    disponiveis = list(df[~df['track_id'].isin(ids_usados)]['track_id'])

    return cromossoma, disponiveis


def crossover(pai1_tuplo, pai2_tuplo, df):
    """
    Crossover ao nível das playlists (ponto de corte entre PL2 e PL3):
      filho1 herda PL1+PL2 do pai1 e PL3+PL4 do pai2
      filho2 herda PL1+PL2 do pai2 e PL3+PL4 do pai1
    Após o cruzamento, aplica reparação de conflitos.
    Devolve dois filhos como tuplos (cromossoma, disponiveis).
    """
    pai1, _ = pai1_tuplo
    pai2, _ = pai2_tuplo

    filho1 = {
        'PL1': pai1['PL1'].copy(), 'PL2': pai1['PL2'].copy(),
        'PL3': pai2['PL3'].copy(), 'PL4': pai2['PL4'].copy(),
    }
    filho2 = {
        'PL1': pai2['PL1'].copy(), 'PL2': pai2['PL2'].copy(),
        'PL3': pai1['PL3'].copy(), 'PL4': pai1['PL4'].copy(),
    }

    filho1, disp1 = reparar_conflitos(filho1, df)
    filho2, disp2 = reparar_conflitos(filho2, df)

    return (filho1, disp1), (filho2, disp2)


# ------------------------------------------------------------------
# Mutação
# ------------------------------------------------------------------

def mutacao(cromossoma_tuplo, df, prob_mutacao):
    """
    Operador de mutação: com probabilidade prob_mutacao, aplica
    um movimento de substituição (reutiliza a vizinhança do SA).
    Devolve o cromossoma (possivelmente mutado) e o pool actualizado.
    """
    cromossoma, disponiveis = cromossoma_tuplo

    if random.random() < prob_mutacao:
        nova_sol, novo_disp, _, sucesso = movimento_substituicao(
            cromossoma, df, disponiveis
        )
        if sucesso:
            return nova_sol, novo_disp

    return cromossoma, disponiveis


# ------------------------------------------------------------------
# Algoritmo Genético principal
# ------------------------------------------------------------------

def algoritmo_genetico(df, n_cromossomas, n_geracoes, prob_mutacao,
                       caminho_output, k_torneio=5):
    """
    Executa o Algoritmo Genético completo.

    Parâmetros:
        df             — DataFrame com todas as músicas
        n_cromossomas  — dimensão da população
        n_geracoes     — número de gerações (critério de paragem)
        prob_mutacao   — probabilidade de mutação (ex: 0.1)
        caminho_output — caminho para o ficheiro CSV de log
        k_torneio      — tamanho do torneio na selecção

    Devolve:
        melhor_solucao — melhor solução encontrada
        melhor_pop     — popularidade total da melhor solução
    """
    print(f"AG iniciado | População: {n_cromossomas} | Gerações: {n_geracoes} | "
          f"P(mutação): {prob_mutacao} | k_torneio: {k_torneio}")

    # --- Geração da população inicial ---
    print("A gerar população inicial...")
    populacao = gerar_populacao_inicial(df, n_cromossomas)
    aptidoes  = [calcular_popularidade(c, df) for c, _ in populacao]

    melhor_idx     = aptidoes.index(max(aptidoes))
    melhor_solucao = copy.deepcopy(populacao[melhor_idx][0])
    melhor_pop     = aptidoes[melhor_idx]

    print(f"População inicial | Melhor: {melhor_pop} | Média: {sum(aptidoes)/len(aptidoes):.1f}\n")

    log = []

    # --- Ciclo de gerações ---
    for geracao in range(1, n_geracoes + 1):

        nova_populacao = []

        # Elitismo: o melhor cromossoma passa directamente para a próxima geração
        nova_populacao.append(copy.deepcopy(populacao[melhor_idx]))

        # Gerar os restantes cromossomas
        while len(nova_populacao) < n_cromossomas:
            idx_pai1 = selecionar_pai(populacao, df, k=k_torneio)
            idx_pai2 = selecionar_pai(populacao, df, k=k_torneio)
            while idx_pai2 == idx_pai1:
                idx_pai2 = selecionar_pai(populacao, df, k=k_torneio)

            filho1_t, filho2_t = crossover(populacao[idx_pai1], populacao[idx_pai2], df)
            filho1_t = mutacao(filho1_t, df, prob_mutacao)
            filho2_t = mutacao(filho2_t, df, prob_mutacao)

            nova_populacao.append(filho1_t)
            if len(nova_populacao) < n_cromossomas:
                nova_populacao.append(filho2_t)

        # Substituição geracional completa
        populacao  = nova_populacao
        aptidoes   = [calcular_popularidade(c, df) for c, _ in populacao]
        melhor_idx = aptidoes.index(max(aptidoes))
        media_apt  = sum(aptidoes) / len(aptidoes)
        pior_apt   = min(aptidoes)

        if aptidoes[melhor_idx] > melhor_pop:
            melhor_pop     = aptidoes[melhor_idx]
            melhor_solucao = copy.deepcopy(populacao[melhor_idx][0])
            print(f"  ✅ Geração {geracao:3d}: Nova melhor solução = {melhor_pop}")

        log.append({
            'geracao'        : geracao,
            'melhor_aptidao' : aptidoes[melhor_idx],
            'media_aptidao'  : round(media_apt, 2),
            'pior_aptidao'   : pior_apt,
            'melhor_global'  : melhor_pop,
        })

        if geracao % 10 == 0:
            print(f"  Geração {geracao:3d} | Melhor: {aptidoes[melhor_idx]} | "
                  f"Média: {media_apt:.1f} | Pior: {pior_apt}")

    # --- Guardar log ---
    df_log = pd.DataFrame(log)
    df_log.to_csv(caminho_output, index=False)
    print(f"\nLog guardado em: {caminho_output}")
    print(f"Melhor solução AG — Popularidade: {melhor_pop}")

    return melhor_solucao, melhor_pop