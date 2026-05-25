import random
random.seed(42)

from src.carrega_dados import carregar_dataset
from src.heuristica_construtiva import heuristica_construtiva
from src.funcao_objetivo import calcular_popularidade, verificar_admissibilidade
from src.vizinhanca import movimento_substituicao

df = carregar_dataset()
solucao, disponiveis = heuristica_construtiva(df)
pop_inicial = calcular_popularidade(solucao, df)
print(f"Solução inicial: {pop_inicial}")

# Aplicar Movimento 1 e mostrar o resultado real
nova_sol, novo_disp, descricao, ok = movimento_substituicao(solucao, df, disponiveis)

if ok:
    pop_nova = calcular_popularidade(nova_sol, df)
    adm, msgs = verificar_admissibilidade(nova_sol, df)
    print(f"Movimento: {descricao}")
    print(f"Popularidade vizinho: {pop_nova} ({pop_nova - pop_inicial:+d})")
    print(f"Admissível: {adm}")
    for msg in msgs:
        print(f"  {msg}")
    
    # Mostrar detalhes da playlist afectada
    pl_afectada = descricao.split()[2].rstrip(":")  # extrai "PL4" por exemplo
    musicas = df[df['track_id'].isin(nova_sol[pl_afectada])]
    print(f"\nComposição da {pl_afectada} após o movimento:")
    for _, row in musicas.iterrows():
        print(f"  {row['track_name']} — pop={row['popularity']} | valence={row['valence']:.3f}")