import random
random.seed(42)

from src.carrega_dados import carregar_dataset
from src.heuristica_construtiva import heuristica_construtiva
from src.funcao_objetivo import calcular_popularidade, verificar_admissibilidade
from src.simulated_annealing import simulated_annealing, M_ENUNCIADO, M_ALTERNATIVO

# ============================================================
# main_sa.py — Script principal do Simulated Annealing
# Executar com: python main_sa.py
# ============================================================

# --- 1. Carregar dados ---
print("A carregar dataset...")
df = carregar_dataset()
print(f"Dataset: {len(df)} músicas\n")

# --- 2. Solução inicial ---
solucao_inicial, musicas_disponiveis = heuristica_construtiva(df)
pop_inicial = calcular_popularidade(solucao_inicial, df)
print(f"Solução inicial (heurística construtiva) — Popularidade: {pop_inicial}\n")

# ---------------------------------------------------------------
# Executar SA com o esquema do enunciado: M = [10,10,10,5,5]
# ---------------------------------------------------------------
print("=" * 60)
print("SIMULATED ANNEALING — Esquema do Enunciado")
print(f"M = {M_ENUNCIADO} | Total iterações: {sum(M_ENUNCIADO)}")
print("=" * 60)

melhor_sol_1, melhor_pop_1 = simulated_annealing(
    solucao_inicial,
    df,
    musicas_disponiveis.copy(),
    caminho_output="output/output_sa_enunciado.csv",
    esquema_M=M_ENUNCIADO
)

adm_1, msgs_1 = verificar_admissibilidade(melhor_sol_1, df)
print(f"\nVerificação de admissibilidade:")
for msg in msgs_1:
    print(f"  {msg}")

# ---------------------------------------------------------------
# Executar SA com esquema alternativo (mais iterações)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("SIMULATED ANNEALING — Esquema Alternativo")
print(f"M = {M_ALTERNATIVO} | Total iterações: {sum(M_ALTERNATIVO)}")
print("=" * 60)

# Reiniciar a partir da mesma solução inicial para comparação justa
random.seed(42)
melhor_sol_2, melhor_pop_2 = simulated_annealing(
    solucao_inicial,
    df,
    musicas_disponiveis.copy(),
    caminho_output="output/output_sa_alternativo.csv",
    esquema_M=M_ALTERNATIVO
)

adm_2, msgs_2 = verificar_admissibilidade(melhor_sol_2, df)
print(f"\nVerificação de admissibilidade:")
for msg in msgs_2:
    print(f"  {msg}")

# ---------------------------------------------------------------
# Comparação final
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMO COMPARATIVO")
print("=" * 60)
print(f"Solução inicial (heurística):     {pop_inicial}")
print(f"SA — Esquema enunciado:           {melhor_pop_1}  ({melhor_pop_1 - pop_inicial:+d})")
print(f"SA — Esquema alternativo:         {melhor_pop_2}  ({melhor_pop_2 - pop_inicial:+d})")

