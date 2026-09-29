# =========================================================
# Bibliotecas
# =========================================================

import warnings
warnings.filterwarnings("ignore")
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, weightedtau
from copy import deepcopy
import subprocess
import time
# =========================================================
# Meus módulos
# =========================================================

from utility_functions import UtilityFunctions
from utastar_with_preferred_alt_groups_online_learning import (UTASTAR)
from multiobjective_knapsack_problem import (MOKP)
from funcoes_auxiliares import *
from debug_utils import debug_print, debug_show
from html_report import HtmlReport
import config_experimento as cfg


# =========================================================
# Confuganro os parâmetros do experimento. Para modificar, ir no arquivo: config_experimento
# =========================================================
seed_experimento = cfg.seed_experimento
seed_utilidade = cfg.seed_utilidade

debug_utastar = cfg.debug_utastar
DEBUG_REPORT = cfg.DEBUG_REPORT
detalhar_solucoes = cfg.detalhar_solucoes
SALVAR_FO_UTASTAR = cfg.SALVAR_FO_UTASTAR
SALVAR_DADOS_ILLUSTRATIVE = cfg.SALVAR_DADOS_ILLUSTRATIVE
SALVAR_RESULTADOS_EXPERIMENTO = cfg.SALVAR_RESULTADOS_EXPERIMENTO

num_fronteiras_pareto = cfg.num_fronteiras_pareto
num_fronteiras_teste = cfg.num_fronteiras_teste
num_execucoes = cfg.num_execucoes
num_objetivos = cfg.num_objetivos
lista_pontos_utastar = cfg.lista_pontos_utastar.copy()
num_pesos_soma_ponderada = cfg.num_pesos_soma_ponderada
percentual_alt_no_grupo_preferido = cfg.percentual_alt_no_grupo_preferido
num_monte_carlo = cfg.num_monte_carlo
num_decisores = cfg.num_decisores
epsilon = cfg.epsilon
tipo_feedback = cfg.tipo_feedback
parametros = deepcopy(cfg.parametros)
tipos_utilidade = cfg.tipos_utilidade.copy()
plot_comparacao_pos_otimizacao = cfg.plot_comparacao_pos_otimizacao

plot_funcoes_utilidade_reais = cfg.plot_funcoes_utilidade_reais

# =========================================================
# Pasta para salvar figuras
# =========================================================

PASTA_FIGURAS = "figuras"
os.makedirs(PASTA_FIGURAS, exist_ok=True)

PASTA_DADOS_ILLUSTRATIVE = "dados_para_illustrative_example"

if SALVAR_DADOS_ILLUSTRATIVE:
    os.makedirs(PASTA_DADOS_ILLUSTRATIVE, exist_ok=True)

# =========================================================
# Código para gerar um report
# =========================================================

relatorio = None
PASTA_RELATORIOS = "relatorios"
nome_experimento = f"exp{seed_experimento}_{tipo_feedback}_util{seed_utilidade}_front{num_fronteiras_pareto}_test{num_fronteiras_teste}_obj{num_objetivos}_lambdas{num_pesos_soma_ponderada}_pref{percentual_alt_no_grupo_preferido}_mc{num_monte_carlo}_dm{num_decisores}"
PASTA_FIGURAS_EXPERIMENTO = os.path.join(PASTA_FIGURAS, nome_experimento)
os.makedirs(PASTA_FIGURAS_EXPERIMENTO, exist_ok=True)
os.makedirs(PASTA_RELATORIOS, exist_ok=True)
nome_relatorio = f"exp{seed_experimento}_{tipo_feedback}_util{seed_utilidade}_front{num_fronteiras_pareto}_test{num_fronteiras_teste}_lambdas{num_pesos_soma_ponderada}_pref{percentual_alt_no_grupo_preferido}_mc{num_monte_carlo}_dm{num_decisores}.html"

if DEBUG_REPORT:
    relatorio = HtmlReport(os.path.join(PASTA_RELATORIOS, nome_relatorio))

# transformar esses parâmetros em tabela
if relatorio:
    relatorio.add_heading("Experiment Configuration", level=1)
    relatorio.add_table(
        ["Parameter", "Value"],
        [["seed_experimento", seed_experimento],
            ["seed_utilidade", seed_utilidade],
            ["N. pareto frontiers (k)", num_fronteiras_pareto],
            ["N. weight sum (alternatives)", num_pesos_soma_ponderada],
            ["Percentage of preferred alternatives", percentual_alt_no_grupo_preferido],
            ["N. of Monte Carlo", num_monte_carlo],
            ["N. decision makers", num_decisores]])
    relatorio.add_hr()

# =========================================================
# Fim do Código para gerar um report
# =========================================================












# =========================================================
# Variável que guarda os Resultados
# =========================================================

resultados_experimentos = {}
valores_fo_utastar = []

dados_resultados_experimento = []
dados_illustrative_config = []
dados_illustrative_pareto = []
dados_illustrative_iteracoes = []
dados_illustrative_ranking = []
dados_illustrative_curvas = []
dados_illustrative_pos_otimizacao = []
dados_illustrative_kendall_historico = []
dados_illustrative_pareto_teste = []
dados_illustrative_kendall_teste = []
dados_tempos_simulacoes = []

if SALVAR_DADOS_ILLUSTRATIVE:
    dados_illustrative_config.extend([
        {"parameter": "seed_experimento", "value": seed_experimento},
        {"parameter": "seed_utilidade", "value": seed_utilidade},
        {"parameter": "tipo_feedback", "value": tipo_feedback},
        {"parameter": "num_fronteiras_pareto", "value": num_fronteiras_pareto},
        {"parameter": "num_fronteiras_teste", "value": num_fronteiras_teste},
        {"parameter": "num_objetivos", "value": num_objetivos},
        {"parameter": "lista_pontos_utastar", "value": str(lista_pontos_utastar)},
        {"parameter": "num_alternativas", "value": num_pesos_soma_ponderada},
        {"parameter": "percentual_grupo_preferido", "value": percentual_alt_no_grupo_preferido},
        {"parameter": "num_monte_carlo", "value": num_monte_carlo},
        {"parameter": "num_decisores", "value": num_decisores},
        {"parameter": "epsilon", "value": epsilon},
        {"parameter": "tipos_utilidade", "value": str(tipos_utilidade)}
    ])

# =========================================================
# Monte Carlo - gera diferentes fronteirs a cada iteração id_mc
# =========================================================

for id_mc in range(num_monte_carlo):

    print("\n================================")
    print(f"MONTE CARLO {id_mc + 1}/{num_monte_carlo}")
    print("================================")

    if relatorio:
        relatorio.add_heading(f"Monte Carlo {id_mc + 1}", level=1)

    # =========================================================
    # Gerar fronteiras - faço um while para gerar um mínimo de 5 soluções, até não conseguir 5, o problema fica tentando gerar
    # =========================================================

    fronteiras = []

    tentativas = 0
    i = 0

    while len(fronteiras) < num_fronteiras_pareto:

        tentativas += 1

        if tentativas > 100:
            raise RuntimeError(
                f"Unable to generate {num_fronteiras_pareto} valid Pareto frontiers after 100 attempts."
            )

        seed_iteracao = seed_experimento + id_mc * 10000 + i
        i += 1

        parametros["seed"] = seed_iteracao

        modelo = MOKP(parametros)
        modelo.gerar_instancia()
        modelo.gerar_fronteira(num_lambdas=num_pesos_soma_ponderada)

        objetivos = modelo.fronteira["objetivos"]

        if len(objetivos) < 5:
            continue

        fronteiras.append({
            "seed": seed_iteracao,
            "objetivos": objetivos
        })

    # =========================================================
    # Limites globais das fronteiras de treinamento
    # =========================================================

    todas_solucoes = np.vstack([
        f["objetivos"]
        for f in fronteiras
    ])

    mins_globais = np.min(todas_solucoes, axis=0)
    maxs_globais = np.max(todas_solucoes, axis=0)

    # =========================================================
    # Gerar fronteiras de teste dentro dos limites de treinamento
    # =========================================================

    fronteiras_teste = []

    tentativas_teste = 0
    i_teste = 0

    while len(fronteiras_teste) < num_fronteiras_teste:

        tentativas_teste += 1

        if tentativas_teste > 100:
            raise RuntimeError(
                f"Unable to generate {num_fronteiras_teste} valid test Pareto frontiers after 100 attempts."
            )

        seed_teste = seed_experimento + id_mc * 10000 + 1000 + i_teste
        i_teste += 1

        parametros["seed"] = seed_teste

        modelo_teste = MOKP(parametros)
        modelo_teste.gerar_instancia()
        modelo_teste.gerar_fronteira(num_lambdas=num_pesos_soma_ponderada)

        objetivos_teste = modelo_teste.fronteira["objetivos"]

        if len(objetivos_teste) < 5:
            continue

        dentro_limites = np.all(
            (objetivos_teste >= mins_globais) &
            (objetivos_teste <= maxs_globais)
        )

        if not dentro_limites:
            continue

        fronteiras_teste.append({
            "seed": seed_teste,
            "objetivos": objetivos_teste
        })



    if SALVAR_DADOS_ILLUSTRATIVE:
        for id_fronteira, fronteira in enumerate(fronteiras):
            for id_alternativa, objetivos_alt in enumerate(fronteira["objetivos"]):
                dados_illustrative_pareto.append(
                    {"monte_carlo": id_mc + 1, "frontier": id_fronteira + 1, "seed": fronteira["seed"],
                     "alternative": id_alternativa + 1,
                     **{f"objective_{j + 1}": objetivos_alt[j] for j in range(num_objetivos)}})

    if SALVAR_DADOS_ILLUSTRATIVE:
        for id_fronteira, fronteira in enumerate(fronteiras_teste):
            for id_alternativa, objetivos_alt in enumerate(fronteira["objetivos"]):
                dados_illustrative_pareto_teste.append(
                    {"monte_carlo": id_mc + 1, "test_frontier": id_fronteira + 1, "seed": fronteira["seed"],
                     "alternative": id_alternativa + 1,
                     **{f"objective_{j + 1}": objetivos_alt[j] for j in range(num_objetivos)}})

    # =========================================================
    # Segurança contra IndexError
    # =========================================================

    num_execucoes_atual = min(num_execucoes, len(fronteiras))

    if relatorio:
        nome_figura = f"fronteiras_pareto_mc{id_mc + 1}.png"

        plotar_fronteiras_pareto(
            fronteiras,
            mins_globais=mins_globais,
            maxs_globais=maxs_globais,
            save_path=os.path.join(PASTA_FIGURAS_EXPERIMENTO, nome_figura)
        )

        relatorio.add_heading("Pareto Frontiers", level=2)
        relatorio.add_text(f"Number of frontiers generated: {len(fronteiras)}")
        relatorio.add_text("The figure below shows all Pareto frontiers generated for this Monte Carlo simulation.")
        relatorio.add_image(f"../figuras/{nome_experimento}/{nome_figura}")
        relatorio.add_heading("Global Limits", level=2)
        relatorio.add_table(
            ["Objective", "Minimum", "Maximum"],
            [["Objective 1", f"{mins_globais[0]:.2f}", f"{maxs_globais[0]:.2f}"],
             ["Objective 2", f"{mins_globais[1]:.2f}", f"{maxs_globais[1]:.2f}"]]
        )
        relatorio.add_hr()






    # =========================================================
    # Diferentes Decision makers
    # =========================================================
    seed_base_decisor = seed_utilidade + id_mc * 10000 # altera o seed para cada simulação montecarlo, todos os decisores serem diferentes

    for id_decisor in range(num_decisores):

        if relatorio:
            relatorio.add_heading(f"Decision Maker {id_decisor + 1}", level=2)



        pesos_reais = UtilityFunctions.gerar_pesos_reais(num_criterios=parametros["num_objetivos"], seed=seed_base_decisor + id_decisor)
        seed_utilidade_atual = (seed_base_decisor + id_decisor)
        debug_print(f"Real weights: {np.round(pesos_reais, 4)}")

        if SALVAR_DADOS_ILLUSTRATIVE:
            for j, peso in enumerate(pesos_reais):
                dados_illustrative_config.append(
                    {"parameter": f"mc{id_mc + 1}_dm{id_decisor + 1}_real_weight_objective_{j + 1}", "value": peso})

        if relatorio:
            relatorio.add_heading("Real Weights", level=3)
            relatorio.add_table(["Objective", "Weight"],[ [f"Objective {i + 1}", f"{peso:.4f}"] for i, peso in enumerate(pesos_reais)])


        # =====================================================
        # Tipo utilidade
        # =====================================================

        for tipo_utilidade in tipos_utilidade:


            if relatorio:
                relatorio.add_heading(f"{tipo_utilidade.capitalize()} Utility", level=3)
                relatorio.add_text("Real utility function used to generate preferences.")
                relatorio.add_heading("Utility Curves", level=4)




            pontos_utilidade = (

                UtilityFunctions.gerar_pontos_utilidade(

                    pesos_reais,

                    tipo_utilidade,

                    mins_globais,

                    maxs_globais,

                    parametros["c_max"],

                    seed=seed_utilidade_atual
                )
            )

            interpoladores = (

                UtilityFunctions.gerar_interpoladores(
                    pontos_utilidade
                )
            )

            nome_prefixo = f"mc{id_mc + 1}_dm{id_decisor + 1}_{tipo_utilidade}"

            if plot_funcoes_utilidade_reais:
                UtilityFunctions.plotar_funcoes_utilidade(interpoladores, pontos_utilidade,
                                                          save_prefix=os.path.join(PASTA_FIGURAS_EXPERIMENTO,
                                                                                   nome_prefixo))

                if relatorio:
                    for j in range(parametros["num_objetivos"]):
                        relatorio.add_image(f"../figuras/{nome_experimento}/{nome_prefixo}_criterio_{j + 1}.png",
                                            largura=500)


            parametros_utilidade = {

                "pesos": pesos_reais,

                "pontos_utilidade":
                    pontos_utilidade,

                "interpoladores":
                    interpoladores,

                "percentual_preferido":
                    percentual_alt_no_grupo_preferido
            }

            # =================================================
            # Quantidade pontos UTASTAR
            # =================================================

            for quantidade_w_para_utastar in (lista_pontos_utastar):

                inicio_simulacao = time.perf_counter()

                if (quantidade_w_para_utastar not in resultados_experimentos):
                    resultados_experimentos[quantidade_w_para_utastar] = {}

                if (tipo_utilidade not in resultados_experimentos[quantidade_w_para_utastar]):
                    resultados_experimentos[quantidade_w_para_utastar][tipo_utilidade] = {
                        "kendall": [],
                        "kendall_weighted": [],
                        "kendall_weighted_historical": [],
                        "kendall_weighted_test": [],
                    }

                # =================================================
                # Históricos
                # =================================================

                historico_metricas = {
                    "kendall": [],
                    "kendall_weighted": [],
                    "kendall_weighted_historical": [],
                    "kendall_weighted_test": [],
                }

                historico_preferencias = []

                historico_modelos = []

                todas_alternativas = []

                todos_objetivos = []

                # =================================================
                # Inicialização
                # =================================================

                pesos_estimados = np.ones(
                    parametros["num_objetivos"]
                )

                pesos_estimados = (
                    pesos_estimados
                    / np.sum(pesos_estimados)
                )

                pontos_estimados = []

                for j in range(
                    parametros["num_objetivos"]
                ):

                    x = np.linspace(

                        mins_globais[j],

                        maxs_globais[j],

                        quantidade_w_para_utastar
                    )

                    y = np.linspace(

                        0,

                        pesos_estimados[j],

                        quantidade_w_para_utastar
                    )

                    pontos_estimados.append({

                        "x": x,

                        "y": y
                    })

                # =================================================
                # Execuções
                # =================================================


                for i in range(num_execucoes_atual):
                    inicio_round = time.perf_counter()





                    # =================================================
                    # para gerar relatório
                    # =================================================
                    if relatorio:
                        relatorio.add_heading(f"Iteration {i + 1}", level=4)

                        nome_before = f"mc{id_mc + 1}_dm{id_decisor + 1}_{tipo_utilidade}_{quantidade_w_para_utastar}p_iter{i + 1}_before_learning.png"

                        plotar_real_vs_estimada_antes_aprendizado(
                            interpoladores,
                            pontos_estimados,
                            save_path=os.path.join(PASTA_FIGURAS_EXPERIMENTO, nome_before)
                        )

                        relatorio.add_heading("Utility Functions Before Learning", level=5)
                        relatorio.add_image(f"../figuras/{nome_experimento}/{nome_before}", largura=900)

                        relatorio.add_heading("Real vs Estimated Weights Before Learning", level=5)
                        relatorio.add_table(
                            ["Criterion", "Real Weight", "Estimated Weight"],
                            [[f"Objective {j + 1}", f"{pesos_reais[j]:.4f}", f"{pesos_estimados[j]:.4f}"] for j in
                             range(parametros["num_objetivos"])] )

                    seed_iteracao = fronteiras[i]["seed"]

                    parametros["seed"] = (
                        seed_iteracao
                    )


                    objetivos = (fronteiras[i]["objetivos"])

                    objetivos_brutos = objetivos

                    # =============================================
                    # Utilidades
                    # =============================================

                    if SALVAR_DADOS_ILLUSTRATIVE or SALVAR_RESULTADOS_EXPERIMENTO:
                        pesos_before_learning = pesos_estimados.copy()



                    utilidades_reais = (

                        UtilityFunctions.avaliar_utilidade(

                            objetivos_brutos,

                            parametros_utilidade,

                            tipo_utilidade
                        )
                    )

                    utilidades_estimadas = (

                        UtilityFunctions.
                        avaliar_utilidade_estimada(

                            objetivos_brutos,

                            pesos_estimados,

                            pontos_estimados
                        )
                    )

                    # =============================================
                    # Rankings
                    # =============================================
                    alternativas = [f"A_{seed_iteracao}_{i+1}_{a}" for a in range(len(objetivos_brutos))]

                    ranking_real = np.argsort(np.argsort(-utilidades_reais))
                    ranking_estimado = np.argsort(np.argsort(-utilidades_estimadas))

                    kendall, _ = kendalltau(ranking_real, ranking_estimado)
                    kendall_weighted, _ = weightedtau(ranking_real, ranking_estimado)

                    kendalls_historicos, kendall_weighted_historico_medio = calcular_weighted_kendall_historico(
                        fronteiras[:i],
                        parametros_utilidade,
                        tipo_utilidade,
                        pesos_estimados,
                        pontos_estimados
                    )

                    kendalls_teste, kendall_weighted_teste_medio = calcular_weighted_kendall_historico(
                        fronteiras_teste,
                        parametros_utilidade,
                        tipo_utilidade,
                        pesos_estimados,
                        pontos_estimados
                    )



                    if SALVAR_DADOS_ILLUSTRATIVE:

                        for s, tau_historico in enumerate(kendalls_historicos):
                            dados_illustrative_kendall_historico.append({
                                "monte_carlo": id_mc + 1,
                                "decision_maker": id_decisor + 1,
                                "utility": tipo_utilidade,
                                "utastar_points": quantidade_w_para_utastar,
                                "iteration": i + 1,
                                "model": f"V_hat_{i}",
                                "historical_frontier": s + 1,
                                "weighted_kendall_historical": tau_historico
                            })

                        for s, tau_teste in enumerate(kendalls_teste):
                            dados_illustrative_kendall_teste.append({
                                "monte_carlo": id_mc + 1,
                                "decision_maker": id_decisor + 1,
                                "utility": tipo_utilidade,
                                "utastar_points": quantidade_w_para_utastar,
                                "iteration": i + 1,
                                "model": f"V_hat_{i}",
                                "test_frontier": s + 1,
                                "weighted_kendall_test": tau_teste
                            })




                    # =============================================
                    # Salvar no relatório:
                    # =============================================

                    # ===== RELATÓRIO: RANKING ANTES DO APRENDIZADO =====
                    if relatorio:
                        ordem_real = np.argsort(-utilidades_reais)
                        ordem_estimada = np.argsort(-utilidades_estimadas)

                        dados_ranking = []

                        for pos in range(len(alternativas)):
                            idx_real = ordem_real[pos]
                            idx_estimado = ordem_estimada[pos]

                            dados_ranking.append([
                                pos + 1,
                                alternativas[idx_real],
                                f"{utilidades_reais[idx_real]:.4f}",
                                alternativas[idx_estimado],
                                f"{utilidades_estimadas[idx_estimado]:.4f}"
                            ])

                        relatorio.add_heading("Real vs Estimated Ranking Before Learning", level=5)
                        relatorio.add_table(
                            ["Position", "Real Alternative", "Real Utility",
                             "Estimated Alternative", "Estimated Utility"],
                            dados_ranking
                        )

                        relatorio.add_heading("Ranking Metrics Before Learning", level=5)
                        relatorio.add_table(["Metric", "Value"], [["Kendall", f"{kendall:.4f}"], ["Weighted Kendall", f"{kendall_weighted:.4f}"]])

                    if relatorio and parametros["num_objetivos"] == 2:
                        nome_paretos_iteracao = f"mc{id_mc + 1}_dm{id_decisor + 1}_{tipo_utilidade}_{quantidade_w_para_utastar}p_iter{i + 1}_paretos_acumuladas.png"
                        plotar_fronteiras_pareto(fronteiras[:i + 1], mins_globais=mins_globais,
                                                 maxs_globais=maxs_globais,
                                                 save_path=os.path.join(PASTA_FIGURAS_EXPERIMENTO,
                                                                        nome_paretos_iteracao))
                        relatorio.add_heading("Pareto Frontiers Used So Far", level=5)
                        relatorio.add_image(f"../figuras/{nome_experimento}/{nome_paretos_iteracao}", largura=700)





                    # =============================================
                    # Histórico métricas
                    # =============================================

                    historico_metricas[
                        "kendall"
                    ].append(kendall)

                    historico_metricas[
                        "kendall_weighted"
                    ].append(kendall_weighted)

                    historico_metricas[
                        "kendall_weighted_historical"
                    ].append(kendall_weighted_historico_medio)

                    historico_metricas[
                        "kendall_weighted_test"
                    ].append(kendall_weighted_teste_medio)




                    # =============================================
                    # Feedback do decisor
                    # =============================================

                    if tipo_feedback == "grupo":
                        grupo_preferido = adicionar_feedback_grupo(historico_preferencias, alternativas,
                                                                   utilidades_reais, percentual_alt_no_grupo_preferido)
                        debug_print(grupo_preferido[0])

                        if relatorio:
                            relatorio.add_heading("Preferred Alternatives", level=5)
                            relatorio.add_table(["Alternative"], [[a] for a in grupo_preferido])

                    elif tipo_feedback == "ranking":
                        ranking = adicionar_feedback_ranking(historico_preferencias, alternativas, utilidades_reais)

                        debug_print(ranking)

                        if relatorio:
                            relatorio.add_heading("Complete Ranking", level=5)
                            relatorio.add_table(["Position", "Alternative"], [[pos + 1, a] for pos, a in enumerate(ranking)])

                    else:
                        raise ValueError(f"Tipo de feedback inválido: {tipo_feedback}")

                    # =================================================
                    # Salva ranking para illustrative example
                    # =================================================

                    if SALVAR_DADOS_ILLUSTRATIVE:

                        for a in range(len(alternativas)):

                            linha_ranking = {
                                "monte_carlo": id_mc + 1,
                                "decision_maker": id_decisor + 1,
                                "utility": tipo_utilidade,
                                "utastar_points": quantidade_w_para_utastar,
                                "iteration": i + 1,
                                "frontier": i + 1,
                                "alternative": alternativas[a],
                                "real_utility": utilidades_reais[a],
                                "estimated_utility_before_learning": utilidades_estimadas[a],
                                "real_rank": int(ranking_real[a]) + 1,
                                "estimated_rank_before_learning": int(ranking_estimado[a]) + 1
                            }

                            for j in range(num_objetivos):
                                linha_ranking[f"objective_{j + 1}"] = objetivos_brutos[a, j]

                            if tipo_feedback == "grupo":
                                linha_ranking["preferred_by_dm"] = alternativas[a] in grupo_preferido
                                linha_ranking["feedback_rank"] = None

                            elif tipo_feedback == "ranking":
                                linha_ranking["preferred_by_dm"] = alternativas[a] == ranking[0]
                                linha_ranking["feedback_rank"] = ranking.index(alternativas[a]) + 1

                            dados_illustrative_ranking.append(linha_ranking)


                    todas_alternativas.extend(alternativas)
                    todos_objetivos.append(objetivos_brutos)
                    matriz_global = np.vstack(todos_objetivos)

                    #-------------------------------------------
                    # Para gerar relatório
                    #-------------------------------------------

                    if relatorio:
                        relatorio.add_heading("Data Entering UTASTAR", level=5)
                        relatorio.add_table(["Measure", "Value"], [["Current frontier alternatives", len(alternativas)], ["Accumulated frontiers", i + 1], ["Accumulated alternatives", len(todas_alternativas)]])

                    mins_iteracao = np.min(objetivos_brutos, axis=0)
                    maxs_iteracao = np.max(objetivos_brutos, axis=0)

                    mins_acumulados = np.min(matriz_global, axis=0)
                    maxs_acumulados = np.max(matriz_global, axis=0)

                    if relatorio:
                        dados_range = []

                        for j in range(parametros["num_objetivos"]):
                            amplitude_global = maxs_globais[j] - mins_globais[j]
                            cobertura_iteracao = (maxs_iteracao[j] - mins_iteracao[
                                j]) / amplitude_global if amplitude_global > 0 else 0
                            cobertura_acumulada = (maxs_acumulados[j] - mins_acumulados[
                                j]) / amplitude_global if amplitude_global > 0 else 0

                            dados_range.append([
                                f"Objective {j + 1}",
                                f"{mins_globais[j]:.4f}",
                                f"{maxs_globais[j]:.4f}",
                                f"{mins_iteracao[j]:.4f}",
                                f"{maxs_iteracao[j]:.4f}",
                                f"{100 * cobertura_iteracao:.1f}%",
                                f"{mins_acumulados[j]:.4f}",
                                f"{maxs_acumulados[j]:.4f}",
                                f"{100 * cobertura_acumulada:.1f}%"
                            ])

                        relatorio.add_heading("Observed Range and Global Scale", level=5)
                        relatorio.add_table(
                            ["Criterion", "Global min", "Global max", "Iteration min", "Iteration max",
                             "Iteration coverage", "Cumulative min", "Cumulative max", "Cumulative coverage"],
                            dados_range
                        )

                    # -------------------------------------------
                    # fim gerar relatório
                    # -------------------------------------------



                    criterios = [f"g{j+1}" for j in range(matriz_global.shape[1])]

                    config_niveis = [{
                            "modo":
                                "interpolated",
                            "num_niveis":
                                quantidade_w_para_utastar
                        }

                        for _ in range(matriz_global.shape[1])]

                    # =============================================
                    # UTASTAR
                    # =============================================

                    modelo_uta = UTASTAR(

                        alternativas= todas_alternativas,

                        criterios= criterios,

                        matriz_decisao= matriz_global,

                        c_max= parametros["c_max"],

                        config_niveis= config_niveis,

                        historico_preferencias=historico_preferencias,

                        mins_globais=mins_globais,

                        maxs_globais=maxs_globais,

                        delta=0.0000001,

                        verbose=debug_utastar,

                        plot_utilidades=False
                    )

                    inicio_utastar = time.perf_counter()
                    modelo_uta.run()
                    utastar_time = time.perf_counter() - inicio_utastar

                    if not modelo_uta.resultado.success:

                        print(
                            f"MC {id_mc + 1}/{num_monte_carlo} | "
                            f"DM {id_decisor + 1} | "
                            f"{tipo_utilidade} | "
                            f"UTASTAR {quantidade_w_para_utastar} | "
                            f"round {i + 1}/{num_execucoes_atual} | FALHOU"
                        )

                        if SALVAR_RESULTADOS_EXPERIMENTO:

                            round_time = time.perf_counter() - inicio_round

                            linha_resultado = {
                                "monte_carlo": id_mc + 1,
                                "decision_maker": id_decisor + 1,
                                "utility": tipo_utilidade,
                                "utastar_points": quantidade_w_para_utastar,
                                "iteration": i + 1,
                                "current_alternatives": len(alternativas),
                                "accumulated_alternatives": len(todas_alternativas),
                                "preference_comparisons": len(modelo_uta.comparacoes),
                                "fo_utastar": np.nan,
                                "weighted_kendall_predictive": kendall_weighted,
                                "weighted_kendall_historical": kendall_weighted_historico_medio,
                                "weighted_kendall_test": kendall_weighted_teste_medio,
                                "utastar_time": utastar_time,
                                "round_time": round_time,
                                "utastar_success": False
                            }

                            for j in range(num_objetivos):
                                linha_resultado[f"real_weight_{j + 1}"] = pesos_reais[j]
                                linha_resultado[f"estimated_weight_{j + 1}"] = pesos_before_learning[j]

                            dados_resultados_experimento.append(linha_resultado)


                        continue

                    if SALVAR_DADOS_ILLUSTRATIVE:
                        fo_original = float(modelo_uta.resultado.fun)

                    if SALVAR_FO_UTASTAR:
                        valores_fo_utastar.append(
                            {"monte_carlo": id_mc + 1, "decision_maker": id_decisor + 1, "utility": tipo_utilidade,
                             "utastar_points": quantidade_w_para_utastar, "iteration": i + 1,
                             "fo": float(modelo_uta.resultado.fun)})
                    #------------------------------------
                    # código para gerar relatório
                    #-------------------------------------
                    if relatorio:
                        dados_niveis = []

                        for j, niveis in enumerate(modelo_uta.niveis):
                            dentro = np.sum((niveis >= mins_acumulados[j]) & (niveis <= maxs_acumulados[j]))

                            dados_niveis.append([
                                f"Objective {j + 1}",
                                ", ".join(f"{x:.4f}" for x in niveis),
                                f"[{mins_acumulados[j]:.4f}, {maxs_acumulados[j]:.4f}]",
                                f"{dentro}/{len(niveis)}"
                            ])

                        relatorio.add_heading("UTASTAR Levels and Observed Range", level=5)
                        relatorio.add_table(["Criterion", "UTASTAR levels", "Observed cumulative range", "Levels inside observed range"], dados_niveis)

                    # ------------------------------------
                    # FIM do código para gerar relatório
                    # -------------------------------------




                    if i == num_execucoes_atual - 1:
                        w_real_projetado = testar_funcao_real_no_utastar(modelo_uta, interpoladores)

                    w_original = modelo_uta.resultado.x[modelo_uta.idx_w].copy()

                    if SALVAR_DADOS_ILLUSTRATIVE:
                        modelo_uta.calcular_pesos_criterios(w_original)
                        pesos_after_utastar = modelo_uta.pesos_criterios.copy()

                    modelo_uta.run_pos_otimizacao(epsilon=epsilon)
                    modelo_uta.calcular_pesos_criterios()

                    if relatorio:
                        nome_after = f"mc{id_mc + 1}_dm{id_decisor + 1}_{tipo_utilidade}_{quantidade_w_para_utastar}p_iter{i + 1}_after_learning.png"

                        plotar_real_vs_estimada_depois_aprendizado(
                            interpoladores,
                            modelo_uta,
                            save_path=os.path.join(
                                PASTA_FIGURAS_EXPERIMENTO,
                                nome_after
                            )
                        )

                        relatorio.add_heading(
                            "Utility Functions After Learning",
                            level=5
                        )

                        relatorio.add_image(
                            f"../figuras/{nome_experimento}/{nome_after}",
                            largura=900
                        )

                    if SALVAR_DADOS_ILLUSTRATIVE:
                        pesos_after_postoptimization = modelo_uta.pesos_criterios.copy()

                        linha_iteracao = {
                            "monte_carlo": id_mc + 1,
                            "decision_maker": id_decisor + 1,
                            "utility": tipo_utilidade,
                            "utastar_points": quantidade_w_para_utastar,
                            "iteration": i + 1,
                            "frontier": i + 1,
                            "current_alternatives": len(alternativas),
                            "accumulated_alternatives": len(todas_alternativas),
                            "kendall_before_learning": kendall,
                            "weighted_kendall_before_learning": kendall_weighted,
                            "weighted_kendall_historical_mean": kendall_weighted_historico_medio,
                            "weighted_kendall_test_mean": kendall_weighted_teste_medio,
                            "fo_utastar": fo_original,
                            "preference_comparisons": len(modelo_uta.comparacoes)
                        }

                        for j in range(num_objetivos):
                            linha_iteracao[f"real_weight_{j + 1}"] = pesos_reais[j]
                            linha_iteracao[f"weight_before_learning_{j + 1}"] = pesos_before_learning[j]
                            linha_iteracao[f"weight_after_utastar_{j + 1}"] = pesos_after_utastar[j]
                            linha_iteracao[f"weight_after_postoptimization_{j + 1}"] = pesos_after_postoptimization[j]

                        dados_illustrative_iteracoes.append(linha_iteracao)

                    if SALVAR_RESULTADOS_EXPERIMENTO:

                        round_time = time.perf_counter() - inicio_round

                        linha_resultado = {
                            "monte_carlo": id_mc + 1,
                            "decision_maker": id_decisor + 1,
                            "utility": tipo_utilidade,
                            "utastar_points": quantidade_w_para_utastar,
                            "iteration": i + 1,
                            "current_alternatives": len(alternativas),
                            "accumulated_alternatives": len(todas_alternativas),
                            "preference_comparisons": len(modelo_uta.comparacoes),
                            "fo_utastar": float(modelo_uta.resultado.fun),
                            "weighted_kendall_predictive": kendall_weighted,
                            "weighted_kendall_historical": kendall_weighted_historico_medio,
                            "weighted_kendall_test": kendall_weighted_teste_medio,
                            "utastar_time": utastar_time,
                            "round_time": round_time,
                            "utastar_success": True,
                        }

                        for j in range(num_objetivos):
                            linha_resultado[f"real_weight_{j + 1}"] = pesos_reais[j]
                            linha_resultado[f"estimated_weight_{j + 1}"] = pesos_before_learning[j]

                        dados_resultados_experimento.append(linha_resultado)

                        print(
                            f"MC {id_mc + 1}/{num_monte_carlo} | "
                            f"DM {id_decisor + 1} | "
                            f"{tipo_utilidade} | "
                            f"UTASTAR {quantidade_w_para_utastar} | "
                            f"round {i + 1}/{num_execucoes_atual} | OK"
                        )




                    # =================================================
                    # Salva curvas para illustrative example
                    # =================================================

                    if SALVAR_DADOS_ILLUSTRATIVE:

                        for j, grupo in enumerate(modelo_uta.grupos_w):

                            x = modelo_uta.niveis[j]

                            # Função de utilidade real nos pontos do UTASTAR
                            y_real = interpoladores[j](x)

                            # Solução original do UTASTAR
                            y_utastar = np.concatenate([
                                [0],
                                np.cumsum(w_original[grupo])
                            ])

                            # Solução após a pós-otimização
                            y_pos = np.concatenate([
                                [0],
                                np.cumsum(modelo_uta.media_geral[grupo])
                            ])

                            for k in range(len(x)):
                                dados_illustrative_curvas.append({
                                    "monte_carlo": id_mc + 1,
                                    "decision_maker": id_decisor + 1,
                                    "utility": tipo_utilidade,
                                    "utastar_points": quantidade_w_para_utastar,
                                    "iteration": i + 1,
                                    "criterion": j + 1,
                                    "point": k + 1,
                                    "x": x[k],
                                    "real_utility": y_real[k],
                                    "utastar_original": y_utastar[k],
                                    "post_optimization": y_pos[k]
                                })



                    # ------------------------------------
                    # código para gerar relatório
                    # -------------------------------------

                    if relatorio:
                        nome_comparacao = f"mc{id_mc + 1}_dm{id_decisor + 1}_{tipo_utilidade}_{quantidade_w_para_utastar}p_iter{i + 1}_utastar.png"

                        plotar_comparacao_pos_otimizacao(interpoladores, modelo_uta, w_original,
                                                         save_path=os.path.join(PASTA_FIGURAS_EXPERIMENTO,
                                                                                nome_comparacao))

                        relatorio.add_heading("UTASTAR Utility Functions", level=5)
                        relatorio.add_image(f"../figuras/{nome_experimento}/{nome_comparacao}", largura=900)

                    # =================================================
                    # Dados da pós-otimização
                    # =================================================

                    z_otimo = modelo_uta.resultado.fun
                    sigma_mais = modelo_uta.resultado.x[modelo_uta.idx_sigma_mais]
                    sigma_menos = modelo_uta.resultado.x[modelo_uta.idx_sigma_menos]
                    erros = sigma_mais + sigma_menos

                    solucoes_pos = np.vstack([
                        modelo_uta.solucoes_max,
                        modelo_uta.solucoes_min
                    ])

                    dados_pos = []

                    for j, grupo in enumerate(modelo_uta.grupos_w):

                        peso_original = np.sum(w_original[grupo])

                        pesos_pos = np.sum(
                            solucoes_pos[:, grupo],
                            axis=1
                        )

                        peso_min = np.min(pesos_pos)
                        peso_max = np.max(pesos_pos)
                        peso_final = modelo_uta.pesos_criterios[j]

                        # =================================================
                        # Salva dados para illustrative example
                        # =================================================

                        if SALVAR_DADOS_ILLUSTRATIVE:
                            dados_illustrative_pos_otimizacao.append({
                                "monte_carlo": id_mc + 1,
                                "decision_maker": id_decisor + 1,
                                "utility": tipo_utilidade,
                                "utastar_points": quantidade_w_para_utastar,
                                "iteration": i + 1,
                                "criterion": j + 1,
                                "fo_utastar": z_otimo,
                                "epsilon": epsilon,
                                "error_limit": z_otimo + epsilon,
                                "preference_comparisons": len(modelo_uta.comparacoes),
                                "total_error": np.sum(erros),
                                "comparisons_with_nonzero_error": int(np.sum(erros > 1e-8)),
                                "original_weight": peso_original,
                                "minimum_weight": peso_min,
                                "maximum_weight": peso_max,
                                "weight_range": peso_max - peso_min,
                                "final_weight": peso_final
                            })

                        # =================================================
                        # Guarda dados para o relatório
                        # =================================================

                        if relatorio:
                            dados_pos.append([
                                f"Objective {j + 1}",
                                f"{peso_original:.4f}",
                                f"{peso_min:.4f}",
                                f"{peso_max:.4f}",
                                f"{peso_max - peso_min:.4f}",
                                f"{peso_final:.4f}"
                            ])

                    # =================================================
                    # Relatório
                    # =================================================

                    if relatorio:
                        relatorio.add_heading(
                            "UTASTAR Optimization",
                            level=5
                        )

                        relatorio.add_table(
                            ["Measure", "Value"],
                            [
                                [
                                    "Original objective value (z*)",
                                    f"{z_otimo:.8f}"
                                ],
                                [
                                    "Post-optimization error limit (z* + epsilon)",
                                    f"{z_otimo + epsilon:.8f}"
                                ],
                                [
                                    "Epsilon",
                                    f"{epsilon:.6f}"
                                ],
                                [
                                    "Preference comparisons",
                                    len(modelo_uta.comparacoes)
                                ],
                                [
                                    "Total error",
                                    f"{np.sum(erros):.8f}"
                                ],
                                [
                                    "Comparisons with non-zero error",
                                    int(np.sum(erros > 1e-8))
                                ]
                            ]
                        )

                        relatorio.add_heading(
                            "Post-optimization Weight Ranges",
                            level=5
                        )

                        relatorio.add_table(
                            [
                                "Criterion",
                                "Original weight",
                                "Minimum",
                                "Maximum",
                                "Range",
                                "Final weight"
                            ],
                            dados_pos
                        )




                    # ------------------------------------
                    # FIM do código para gerar relatório
                    # -------------------------------------









                    if plot_comparacao_pos_otimizacao and i == num_execucoes_atual - 1:
                        plotar_comparacao_pos_otimizacao(
                            interpoladores,
                            modelo_uta,
                            w_original,
                            save_path=os.path.join(PASTA_FIGURAS_EXPERIMENTO,
                                                   f"comparacao_pos_{tipo_utilidade}_{quantidade_w_para_utastar}.png")
                        )


                    if relatorio:
                        relatorio.add_heading("Estimated Weights", level=5)
                        relatorio.add_table(["Objective", "Weight"], [[f"Objective {j + 1}", f"{peso:.4f}"] for j, peso in enumerate(modelo_uta.pesos_criterios)])



                    # =============================================
                    # Guarda modelo
                    # =============================================

                    historico_modelos.append(
                        modelo_uta
                    )

                    # =============================================
                    # Atualiza pesos
                    # =============================================

                    pesos_estimados = (
                        modelo_uta.
                        pesos_criterios.copy()
                    )

                    novos_pontos = []

                    for j, grupo in enumerate(
                        modelo_uta.grupos_w
                    ):

                        w_j = (
                            modelo_uta.media_geral[
                                grupo
                            ]
                        )

                        y = np.concatenate([

                            [0],

                            np.cumsum(w_j)
                        ])

                        x = modelo_uta.niveis[j]

                        novos_pontos.append({

                            "x": x,

                            "y": y
                        })

                    pontos_estimados = (
                        novos_pontos
                    )

                # =================================================
                # Guarda métricas
                # =================================================

                for (
                    nome_metrica,
                    valores
                ) in historico_metricas.items():

                    resultados_experimentos[
                        quantidade_w_para_utastar
                    ][
                        tipo_utilidade
                    ][
                        nome_metrica
                    ].append(valores)



                if SALVAR_RESULTADOS_EXPERIMENTO:
                    simulation_time = time.perf_counter() - inicio_simulacao

                    dados_tempos_simulacoes.append({
                        "monte_carlo": id_mc + 1,
                        "decision_maker": id_decisor + 1,
                        "utility": tipo_utilidade,
                        "utastar_points": quantidade_w_para_utastar,
                        "simulation_time": simulation_time
                    })




# =========================================================
# Salvar relatório
# =========================================================

if relatorio:
    relatorio.save()
    print("RELATORIO SALVO")

    caminho_html = os.path.abspath(os.path.join(PASTA_RELATORIOS, nome_relatorio))
    caminho_pdf = os.path.splitext(caminho_html)[0] + ".pdf"

    chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

    subprocess.run([chrome, "--headless", "--disable-gpu", f"--print-to-pdf={caminho_pdf}", caminho_html], check=True)

    print(f"PDF SALVO: {caminho_pdf}")



# =========================================================
# Salvar análise da função objetivo do UTASTAR
# =========================================================

if SALVAR_FO_UTASTAR and valores_fo_utastar:

    # =====================================================
    # DataFrame com todos os valores individuais da FO
    # =====================================================
    df_fo = pd.DataFrame(valores_fo_utastar)

    # Valores muito pequenos são considerados zero
    tolerancia_zero = 1e-8

    df_fo["fo_zero"] = np.abs(df_fo["fo"]) <= tolerancia_zero
    df_fo["fo_maior_zero"] = ~df_fo["fo_zero"]

    # =====================================================
    # Salvar todos os valores individuais
    # =====================================================
    caminho_fo = os.path.join(
        PASTA_RELATORIOS,
        f"{nome_experimento}_fo_utastar.csv"
    )

    df_fo.to_csv(caminho_fo, index=False)

    # =====================================================
    # Resumo por tipo de utilidade e número de pontos
    # =====================================================

    resumo_fo = (
        df_fo
        .groupby(["utility", "utastar_points"])
        .agg(
            total=("fo", "size"),

            fo_zero=("fo_zero", "sum"),
            fo_maior_zero=("fo_maior_zero", "sum"),

            media_fo=("fo", "mean"),
            minimo_fo=("fo", "min"),
            maximo_fo=("fo", "max"),
            desvio_padrao_fo=("fo", "std")
        )
        .reset_index()
    )

    # Percentuais
    resumo_fo["percentual_zero"] = (
        100 * resumo_fo["fo_zero"] / resumo_fo["total"]
    )

    resumo_fo["percentual_maior_zero"] = (
        100 * resumo_fo["fo_maior_zero"] / resumo_fo["total"]
    )

    # =====================================================
    # Estatísticas considerando SOMENTE FO > 0
    # =====================================================

    df_fo_positiva = df_fo.loc[df_fo["fo_maior_zero"]].copy()

    resumo_positivas = (
        df_fo_positiva
        .groupby(["utility", "utastar_points"])
        .agg(
            media_fo_positiva=("fo", "mean"),
            mediana_fo_positiva=("fo", "median"),
            minimo_fo_positiva=("fo", "min"),
            maximo_fo_positiva=("fo", "max"),
            desvio_padrao_fo_positiva=("fo", "std")
        )
        .reset_index()
    )

    # Junta as estatísticas
    resumo_fo = resumo_fo.merge(
        resumo_positivas,
        on=["utility", "utastar_points"],
        how="left"
    )

    # Se não existir nenhuma FO > 0, coloca zero
    colunas_positivas = [
        "media_fo_positiva",
        "mediana_fo_positiva",
        "minimo_fo_positiva",
        "maximo_fo_positiva",
        "desvio_padrao_fo_positiva"
    ]

    resumo_fo[colunas_positivas] = (
        resumo_fo[colunas_positivas].fillna(0)
    )

    # =====================================================
    # Ordenar tabela
    # =====================================================

    ordem_utilidade = {
        "linear": 0,
        "convexa": 1
    }

    resumo_fo["_ordem"] = resumo_fo["utility"].map(ordem_utilidade)

    resumo_fo = (
        resumo_fo
        .sort_values(["_ordem", "utastar_points"])
        .drop(columns="_ordem")
        .reset_index(drop=True)
    )

    # =====================================================
    # Salvar resumo em CSV
    # =====================================================

    caminho_resumo = os.path.join(
        PASTA_RELATORIOS,
        f"{nome_experimento}_resumo_fo_utastar.csv"
    )

    resumo_fo.to_csv(caminho_resumo, index=False)

    # =====================================================
    # Imprimir tabela bonita no console
    # =====================================================

    print("\n")
    print("=" * 130)
    print("RESUMO DA FUNÇÃO OBJETIVO DO UTASTAR")
    print("=" * 130)

    tabela_console = resumo_fo[
        [
            "utility",
            "utastar_points",
            "total",
            "fo_zero",
            "fo_maior_zero",
            "percentual_maior_zero",
            "media_fo",
            "minimo_fo",
            "maximo_fo",
            "desvio_padrao_fo"
        ]
    ].copy()

    tabela_console.columns = [
        "Utilidade",
        "Pontos",
        "Casos",
        "FO = 0",
        "FO > 0",
        "% FO > 0",
        "FO média",
        "FO mínima",
        "FO máxima",
        "Desvio padrão"
    ]

    print(
        tabela_console.to_string(
            index=False,
            formatters={
                "% FO > 0": lambda x: f"{x:.2f}%",
                "FO média": lambda x: f"{x:.10f}",
                "FO mínima": lambda x: f"{x:.10f}",
                "FO máxima": lambda x: f"{x:.10f}",
                "Desvio padrão": lambda x: f"{x:.10f}"
            }
        )
    )

    # =====================================================
    # Resuminho automático
    # =====================================================

    print("\n")
    print("=" * 130)
    print("RESUMO INTERPRETATIVO")
    print("=" * 130)

    for _, linha in resumo_fo.iterrows():

        utilidade = linha["utility"]
        pontos = int(linha["utastar_points"])

        total = int(linha["total"])
        n_zero = int(linha["fo_zero"])
        n_positiva = int(linha["fo_maior_zero"])

        perc_positiva = linha["percentual_maior_zero"]

        media = linha["media_fo"]
        minimo = linha["minimo_fo"]
        maximo = linha["maximo_fo"]
        desvio = linha["desvio_padrao_fo"]

        print(
            f"\n{utilidade.upper()} - {pontos} pontos UTASTAR:"
        )

        if n_positiva == 0:

            print(
                f"  A FO foi igual a zero em todos os {total} casos "
                f"(100%). Isso indica ajuste perfeito das preferências "
                f"nas execuções analisadas."
            )

        else:

            print(
                f"  FO diferente de zero em {n_positiva} de {total} casos "
                f"({perc_positiva:.2f}%)."
            )

            print(
                f"  Considerando todas as execuções: "
                f"média = {media:.6f}, "
                f"mínimo = {minimo:.6f}, "
                f"máximo = {maximo:.6f}, "
                f"desvio padrão = {desvio:.6f}."
            )

            print(
                f"  Considerando somente FO > 0: "
                f"média = {linha['media_fo_positiva']:.6f}, "
                f"mediana = {linha['mediana_fo_positiva']:.6f}, "
                f"mínimo = {linha['minimo_fo_positiva']:.6f}, "
                f"máximo = {linha['maximo_fo_positiva']:.6f}, "
                f"desvio padrão = "
                f"{linha['desvio_padrao_fo_positiva']:.6f}."
            )

    print("\n" + "=" * 130)
    print(f"Valores individuais salvos em: {caminho_fo}")
    print(f"Resumo salvo em: {caminho_resumo}")
    print("=" * 130)

# =========================================================
# Salvar resultados do experimento
# =========================================================

if SALVAR_RESULTADOS_EXPERIMENTO and dados_resultados_experimento:

    df_resultados = pd.DataFrame(dados_resultados_experimento)

    caminho_resultados = os.path.join(
        PASTA_RELATORIOS,
        f"{nome_experimento}_resultados.csv"
    )

    df_resultados.to_csv(caminho_resultados, index=False)

    print("\n============================================")
    print("RESULTADOS DO EXPERIMENTO SALVOS")
    print("============================================")
    print(f"Linhas salvas: {len(df_resultados)}")
    print(f"Arquivo: {caminho_resultados}")


if SALVAR_DADOS_ILLUSTRATIVE:
    caminho_illustrative = os.path.join(PASTA_DADOS_ILLUSTRATIVE, f"{nome_experimento}_illustrative_example.xlsx")

    with pd.ExcelWriter(caminho_illustrative, engine="openpyxl") as writer:
        pd.DataFrame(dados_illustrative_config).to_excel(writer, sheet_name="configuration", index=False)
        pd.DataFrame(dados_illustrative_pareto).to_excel(writer, sheet_name="pareto", index=False)
        pd.DataFrame(dados_illustrative_pareto_teste).to_excel(writer, sheet_name="test_pareto", index=False)
        pd.DataFrame(dados_illustrative_iteracoes).to_excel(writer, sheet_name="iterations", index=False)
        pd.DataFrame(dados_illustrative_ranking).to_excel(writer, sheet_name="ranking", index=False)
        pd.DataFrame(dados_illustrative_curvas).to_excel(writer, sheet_name="utility_curves", index=False)
        pd.DataFrame(dados_illustrative_pos_otimizacao).to_excel(writer, sheet_name="post_optimization", index=False)
        pd.DataFrame(dados_illustrative_kendall_historico).to_excel(writer, sheet_name="historical_kendall",
                                                                    index=False)
        pd.DataFrame(dados_illustrative_kendall_teste).to_excel(writer, sheet_name="test_kendall", index=False)

    print(f"DADOS DO ILLUSTRATIVE EXAMPLE SALVOS: {caminho_illustrative}")

# =========================================================
# Salvar tempos das simulações
# =========================================================

if SALVAR_RESULTADOS_EXPERIMENTO and dados_tempos_simulacoes:

    df_tempos = pd.DataFrame(dados_tempos_simulacoes)

    caminho_tempos = os.path.join(
        PASTA_RELATORIOS,
        f"{nome_experimento}_tempos_simulacoes.csv"
    )

    df_tempos.to_csv(caminho_tempos, index=False)

    print(f"Tempos das simulações: {caminho_tempos}")