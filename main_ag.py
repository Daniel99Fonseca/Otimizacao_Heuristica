from src.carrega_dados import carregar_dataset
from src.algoritmo_genetico import algoritmo_genetico
from src.funcao_objetivo import calcular_popularidade, verificar_admissibilidade

# ============================================================
# main_ag.py — Script principal do Algoritmo Genético
# Executar com: python main_ag.py
# ============================================================

CAMINHO_OUTPUT  = "output/output_ag.csv"
N_CROMOSSOMAS   = 20
N_GERACOES      = 50
PROB_MUTACAO    = 0.1

if __name__ == "__main__":

    # 1. Carregar dados
    df = carregar_dataset()
    print(f"Dataset carregado: {len(df)} músicas")

    # 2. Correr o Algoritmo Genético
    melhor_solucao, melhor_pop = algoritmo_genetico(
        df, N_CROMOSSOMAS, N_GERACOES, PROB_MUTACAO, CAMINHO_OUTPUT
    )

    adm, msg = verificar_admissibilidade(melhor_solucao, df)
    print(f"\nMelhor solução AG — Popularidade total: {melhor_pop}")
    print(f"Admissível: {adm} | {msg}")
    print(f"Log guardado em: {CAMINHO_OUTPUT}")
