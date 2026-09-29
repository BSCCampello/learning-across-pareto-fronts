# =========================================================
# Seeds
# =========================================================

#seed_experimento = 3 #esses foram usados para o experimento do ilustrative example
#seed_utilidade = 7  #esses foram usados para o experimento do ilustrative example


seed_experimento = 3
seed_utilidade = 7


# =========================================================
# Debug, relatório e gráficos de diagnóstico
# =========================================================

SALVAR_RESULTADOS_EXPERIMENTO = True  # True: salva os resultados da simulação; False: não salva nada


SALVAR_DADOS_ILLUSTRATIVE = False  # True: salva os dados do illustrative example em Excel
debug_utastar = False          # True: imprime no console os detalhes internos do UTASTAR
DEBUG_REPORT = False            # True: gera o relatório HTML
detalhar_solucoes = False      # True: mostra detalhes adicionais das soluções
plot_comparacao_pos = False    # True: gera Real x UTASTAR original x pós-otimização
plot_evolucao_utilidade = False # True: gera curvas de utilidade ao longo das iterações
plot_metricas = False           # True: gera gráficos de Kendall
plot_funcoes_utilidade_reais = False
plot_comparacao_pos_otimizacao = False
SALVAR_FO_UTASTAR = False


# =========================================================
# Configuração do experimento
# =========================================================

tipo_feedback = "grupo"  # "ranking" ou "grupo" se for grupo, o utastar recebe apenas o grupo de preferidas do decisor, se for ranking, recebe o ranking completo
num_pesos_soma_ponderada = 50  #numero de alternativas
num_objetivos = 2
percentual_alt_no_grupo_preferido = 0.3


lista_pontos_utastar = [2, 4]
num_fronteiras_pareto = 20 #numero vezes que o decisor mostra suas preferencias, é o que chamamos de k
num_fronteiras_teste = 10 #numero de fronteiras que vão ser geradas mas que não vão entrar no treinamento, são apenas para avaliar o modelo

num_monte_carlo = 300
num_decisores = 1
epsilon = 1e-2
num_execucoes = num_fronteiras_pareto

# =========================================================
# Parâmetros do problema MOKP
# =========================================================

parametros = {
    "num_itens": 120,
    "num_objetivos": num_objetivos,
    "correlacao_objetivos": 0.6,
    "capacidade_min": 0.10,
    "capacidade_max": 0.40,
    "peso_min": 1,
    "peso_max": 80,
    "lucro_min": 1,
    "lucro_max": 300,
    "seed": None
}

parametros["c_max"] = [True] * num_objetivos


# =========================================================
# Tipos de função de utilidade
# =========================================================

tipos_utilidade = ["linear", "convexa"]