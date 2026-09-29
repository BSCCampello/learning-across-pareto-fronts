import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Pastas
# =========================================================

PASTA_ENTRADA = os.path.join(
    "relatorios",
    "rodados e guardados"
)

PASTA_ANALISE = "analise_resultados"

PASTA_GRAFICOS = os.path.join(
    PASTA_ANALISE,
    "graficos_kendall"
)

PASTA_TABELAS = os.path.join(
    PASTA_ANALISE,
    "tabelas"
)

os.makedirs(PASTA_GRAFICOS, exist_ok=True)
os.makedirs(PASTA_TABELAS, exist_ok=True)

# =========================================================
# Encontrar arquivos automaticamente
# =========================================================

todos_arquivos = glob.glob(
    os.path.join(PASTA_ENTRADA, "*.csv")
)

arquivos_resultados = {}
arquivos_tempos = {}

for caminho in todos_arquivos:

    nome = os.path.basename(caminho)

    if nome.endswith("_resultados.csv"):

        nome_experimento = nome.removesuffix(
            "_resultados.csv"
        )

        arquivos_resultados[nome_experimento] = caminho

    elif nome.endswith("_tempos_simulacoes.csv"):

        nome_experimento = nome.removesuffix(
            "_tempos_simulacoes.csv"
        )

        arquivos_tempos[nome_experimento] = caminho

print("\n============================================")
print("ARQUIVOS ENCONTRADOS")
print("============================================")

print(
    f"Arquivos de resultados: "
    f"{len(arquivos_resultados)}"
)

print(
    f"Arquivos de tempos: "
    f"{len(arquivos_tempos)}"
)

# =========================================================
# Métrica Kendall
# =========================================================

metricas = {
    "weighted_kendall_predictive":
        "Predictive weighted Kendall"
}

# =========================================================
# Percorrer experimentos
# =========================================================

for nome_experimento, caminho_resultados in arquivos_resultados.items():

    print("\n============================================")
    print(nome_experimento)
    print("============================================")

    df = pd.read_csv(caminho_resultados)

    # =====================================================
    # Pasta específica do experimento
    # =====================================================

    # Nome curto para a pasta do experimento
    partes = nome_experimento.split("_")

    pref = next((x for x in partes if x.startswith("pref")), "pref")
    lambdas = next((x for x in partes if x.startswith("lambdas")), "lambdas")
    obj = next((x for x in partes if x.startswith("obj")), "obj")

    if "_ranking_" in nome_experimento:
        nome_curto = f"{obj}_{lambdas}_ranking"
    else:
        nome_curto = f"{obj}_{lambdas}_grupo_{pref}"


    # =====================================================
    # Auditoria
    # =====================================================

    print(f"Linhas: {len(df)}")

    print(
        f"Monte Carlos: "
        f"{df['monte_carlo'].nunique()}"
    )

    print(
        f"Decision makers: "
        f"{df['decision_maker'].nunique()}"
    )

    print(
        f"Utilidades: "
        f"{sorted(df['utility'].unique())}"
    )

    print(
        f"UTASTAR points: "
        f"{sorted(df['utastar_points'].unique())}"
    )

    print(
        f"Rounds: "
        f"{df['iteration'].min()} "
        f"a "
        f"{df['iteration'].max()}"
    )

    print(
        f"Sucessos: "
        f"{int(df['utastar_success'].sum())}"
    )

    print(
        f"Falhas: "
        f"{int((~df['utastar_success']).sum())}"
    )

    # =====================================================
    # Erro dos pesos
    # =====================================================

    colunas_reais = sorted(
        [
            c for c in df.columns
            if c.startswith("real_weight_")
        ],
        key=lambda x: int(x.split("_")[-1])
    )

    colunas_estimadas = sorted(
        [
            c for c in df.columns
            if c.startswith("estimated_weight_")
        ],
        key=lambda x: int(x.split("_")[-1])
    )

    erros_pesos = []

    for real, estimado in zip(
        colunas_reais,
        colunas_estimadas
    ):

        numero = real.split("_")[-1]

        coluna_erro = (
            f"abs_error_weight_{numero}"
        )

        df[coluna_erro] = np.abs(
            df[real] - df[estimado]
        )

        erros_pesos.append(
            coluna_erro
        )

    df["weight_mae"] = (
        df[erros_pesos]
        .mean(axis=1)
    )

    # =====================================================
    # Resumo por round
    # =====================================================

    resumo_round = (
        df
        .groupby(
            [
                "utility",
                "utastar_points",
                "iteration"
            ]
        )
        .agg(

            n=(
                "weighted_kendall_predictive",
                "size"
            ),

            kendall_predictive_mean=(
                "weighted_kendall_predictive",
                "mean"
            ),

            kendall_predictive_std=(
                "weighted_kendall_predictive",
                "std"
            ),

            weight_mae_mean=(
                "weight_mae",
                "mean"
            ),

            weight_mae_std=(
                "weight_mae",
                "std"
            ),

            fo_mean=(
                "fo_utastar",
                "mean"
            ),

            fo_median=(
                "fo_utastar",
                "median"
            ),

            fo_max=(
                "fo_utastar",
                "max"
            ),

            comparisons_mean=(
                "preference_comparisons",
                "mean"
            ),

            current_alternatives_mean=(
                "current_alternatives",
                "mean"
            ),

            accumulated_alternatives_mean=(
                "accumulated_alternatives",
                "mean"
            ),

            utastar_time_mean=(
                "utastar_time",
                "mean"
            ),

            round_time_mean=(
                "round_time",
                "mean"
            )
        )
        .reset_index()
    )

    resumo_round.to_csv(
        os.path.join(
            PASTA_TABELAS,
            f"{nome_experimento}_resumo_por_round.csv"
        ),
        index=False
    )

    # =====================================================
    # Gráfico do weighted Kendall predictive
    # =====================================================

    for metrica, titulo_metrica in metricas.items():

        for tipo_utilidade in sorted(
            df["utility"].unique()
        ):

            dados = df[
                df["utility"]
                == tipo_utilidade
            ]

            plt.figure(
                figsize=(8, 5)
            )

            for pontos in sorted(
                dados[
                    "utastar_points"
                ].unique()
            ):

                dados_pontos = dados[
                    dados["utastar_points"]
                    == pontos
                ]

                resumo = (
                    dados_pontos
                    .groupby("iteration")[metrica]
                    .agg(
                        [
                            "mean",
                            "std",
                            "count"
                        ]
                    )
                    .reset_index()
                )

                resumo["se"] = (
                    resumo["std"]
                    / np.sqrt(
                        resumo["count"]
                    )
                )

                resumo["ic95"] = (
                    1.96
                    * resumo["se"]
                )

                linha = plt.plot(
                    resumo["iteration"],
                    resumo["mean"],
                    marker="o",
                    label=(
                        f"{pontos} "
                        f"characteristic points"
                    )
                )[0]

                plt.fill_between(
                    resumo["iteration"],
                    resumo["mean"]
                    - resumo["ic95"],
                    resumo["mean"]
                    + resumo["ic95"],
                    alpha=0.15,
                    color=linha.get_color()
                )

            plt.xlabel(
                "Decision round"
            )

            plt.ylabel(
                r"$\tau_w$"
            )

            plt.xticks(
                range(1, 21)
            )

            plt.ylim(
                -1,
                1
            )

            plt.grid(
                alpha=0.3
            )

            plt.legend()

            plt.tight_layout()

            if "_ranking_" in nome_experimento:
                nome_figura = (
                    f"{metrica}_"
                    f"{tipo_utilidade}_"
                    f"{obj}_"
                    f"{lambdas}_"
                    f"ranking.png"
                )
            else:
                nome_figura = (
                    f"{metrica}_"
                    f"{tipo_utilidade}_"
                    f"{obj}_"
                    f"{lambdas}_"
                    f"grupo_"
                    f"{pref}.png"
                )

            plt.savefig(
                os.path.join(
                    PASTA_GRAFICOS,
                    nome_figura
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

    # =====================================================
    # Arquivo de tempos correspondente
    # =====================================================

    if nome_experimento in arquivos_tempos:

        caminho_tempos = (
            arquivos_tempos[
                nome_experimento
            ]
        )

        df_tempos = pd.read_csv(
            caminho_tempos
        )

        resumo_tempos = (
            df_tempos
            .groupby(
                [
                    "utility",
                    "utastar_points"
                ]
            )
            .agg(

                n=(
                    "simulation_time",
                    "size"
                ),

                simulation_time_mean=(
                    "simulation_time",
                    "mean"
                ),

                simulation_time_std=(
                    "simulation_time",
                    "std"
                ),

                simulation_time_min=(
                    "simulation_time",
                    "min"
                ),

                simulation_time_max=(
                    "simulation_time",
                    "max"
                )
            )
            .reset_index()
        )

        resumo_tempos.to_csv(
            os.path.join(
                PASTA_TABELAS,
                f"{nome_experimento}_resumo_tempos.csv"
            ),
            index=False
        )

        print(
            "Arquivo de tempos encontrado."
        )

    else:

        print(
            "ATENÇÃO: arquivo de tempos "
            "não encontrado para este experimento."
        )

    print(
        "Gráficos e tabelas gerados."
    )

print("\n============================================")
print("ANÁLISE FINALIZADA")
print("============================================")