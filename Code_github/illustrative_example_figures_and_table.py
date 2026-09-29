# ============================================================
# ILLUSTRATIVE EXAMPLE - FIGURES AND TABLES
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PASTA_DADOS = "dados_para_illustrative_example"

NOME_ARQUIVO = (
    "exp3_grupo_util7_front5_test2_lambdas10_"
    "pref0.1_mc1_dm1_illustrative_example.xlsx"
)

ARQUIVO_EXCEL = os.path.join(
    PASTA_DADOS,
    NOME_ARQUIVO
)

PASTA_FIGURAS = "figuras_illustrative_example"

os.makedirs(
    PASTA_FIGURAS,
    exist_ok=True
)


# ============================================================
# READ DATA
# ============================================================

df_config = pd.read_excel(
    ARQUIVO_EXCEL,
    sheet_name="configuration"
)

df_pareto = pd.read_excel(
    ARQUIVO_EXCEL,
    sheet_name="pareto"
)

df_iterations = pd.read_excel(
    ARQUIVO_EXCEL,
    sheet_name="iterations"
)

df_ranking = pd.read_excel(
    ARQUIVO_EXCEL,
    sheet_name="ranking"
)

df_curves = pd.read_excel(
    ARQUIVO_EXCEL,
    sheet_name="utility_curves"
)

df_post = pd.read_excel(
    ARQUIVO_EXCEL,
    sheet_name="post_optimization"
)


# ============================================================
# BASIC CHECK
# ============================================================

print("\n==============================")
print("ILLUSTRATIVE EXAMPLE")
print("==============================")

print(f"Configuration:      {len(df_config)} rows")
print(f"Pareto:             {len(df_pareto)} rows")
print(f"Iterations:         {len(df_iterations)} rows")
print(f"Ranking:            {len(df_ranking)} rows")
print(f"Utility curves:     {len(df_curves)} rows")
print(f"Post-optimization:  {len(df_post)} rows")


# ============================================================
# FIGURE 1 - PARETO FRONTIERS
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

for fronteira in sorted(
    df_pareto["frontier"].unique()
):

    dados = df_pareto[
        df_pareto["frontier"] == fronteira
    ].copy()

    dados = dados.sort_values("objective_1")

    ax.plot(
        dados["objective_1"],
        dados["objective_2"],
        marker="o",
        markersize=6,
        linewidth=1.5,
        label=f"Frontier {fronteira}"
    )


ax.set_xlabel(
    "Objective 1"
)

ax.set_ylabel(
    "Objective 2"
)

ax.legend(
    frameon=False
)

ax.grid(
    alpha=0.2
)

fig.tight_layout()


# ============================================================
# SAVE FIGURE
# ============================================================

caminho_png = os.path.join(
    PASTA_FIGURAS,
    "illustrative_pareto_frontiers.png"
)

caminho_pdf = os.path.join(
    PASTA_FIGURAS,
    "illustrative_pareto_frontiers.pdf"
)

fig.savefig(
    caminho_png,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    caminho_pdf,
    bbox_inches="tight"
)

plt.show()


print("\nFigure saved:")
print(caminho_png)
print(caminho_pdf)


# ============================================================
# FIGURE 2 - TRUE MARGINAL VALUE FUNCTIONS
# ============================================================

dados = df_curves[
    (df_curves["utility"] == "convexa") &
    (df_curves["utastar_points"] == 4) &
    (df_curves["iteration"] == 1)
].copy()

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

for j, ax in enumerate(axes, start=1):

    criterio = dados[dados["criterion"] == j].sort_values("x")

    ax.plot(
        criterio["x"],
        criterio["real_utility"],
        marker="o",
        markersize=5.5,
        linewidth=1.5
    )

    ax.set_xlabel(f"Objective {j} value")
    ax.set_ylabel("Marginal value")

    ax.set_ylim(0, 1)

    ax.grid(True, which="major", linewidth=0.6, alpha=0.18)
    ax.set_axisbelow(True)

fig.tight_layout()

fig.savefig(
    os.path.join(
        PASTA_FIGURAS,
        "illustrative_true_value_functions.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()



# ============================================================
# TABLE 1 - LEARNING EVOLUTION
# ============================================================

tabela_aprendizado = df_iterations[
    (df_iterations["utility"] == "convexa") &
    (df_iterations["utastar_points"] == 4)
].copy()

tabela_aprendizado = tabela_aprendizado.sort_values("iteration")

tabela_aprendizado = tabela_aprendizado[
    [
        "iteration",
        "preference_comparisons",
        "fo_utastar",
        "weight_before_learning_1",
        "weight_before_learning_2",
        "weighted_kendall_before_learning",
        "weighted_kendall_historical_mean",
        "weighted_kendall_test_mean"
    ]
].copy()

print("\nTABLE 1 - LEARNING EVOLUTION")

print(r"\begin{table}[ht]")
print(r"\centering")
print(r"\caption{Evolution of the preference learning process in the illustrative example using four UTASTAR characteristic points.}")
print(r"\label{tab:illustrative_learning_evolution}")
print(r"\begin{tabular}{cccccccc}")
print(r"\hline")
print(r"$t$ & Comparisons & $z^*$ & $\hat{w}_{1,t-1}$ & $\hat{w}_{2,t-1}$ & $\tau_{w,t}^{pred}$ & $\bar{\tau}_{w,t}^{hist}$ & $\bar{\tau}_{w,t}^{test}$ \\")
print(r"\hline")

for _, linha in tabela_aprendizado.iterrows():

    kendall_historico = (
        "--"
        if pd.isna(linha["weighted_kendall_historical_mean"])
        else f"{linha['weighted_kendall_historical_mean']:.4f}"
    )

    print(
        f"{int(linha['iteration'])} & "
        f"{int(linha['preference_comparisons'])} & "
        f"{linha['fo_utastar']:.0f} & "
        f"{linha['weight_before_learning_1']:.4f} & "
        f"{linha['weight_before_learning_2']:.4f} & "
        f"{linha['weighted_kendall_before_learning']:.4f} & "
        f"{kendall_historico} & "
        f"{linha['weighted_kendall_test_mean']:.4f} \\\\"
    )

print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")




# ============================================================
# FIGURE 3 - RECOVERY OF MARGINAL VALUE FUNCTIONS
# ============================================================

dados_recuperacao = df_curves[
    (df_curves["utility"] == "convexa") &
    (df_curves["utastar_points"] == 4)
].copy()

rounds_mostrados = [1, 3, 5]

fig, axes = plt.subplots(
    3, 2,
    figsize=(10, 8.5),
    sharey=True
)

# Mantém a mesma escala horizontal em todos os rounds
limites_x = {}

for criterio in [1, 2]:
    dados_criterio = dados_recuperacao[
        dados_recuperacao["criterion"] == criterio
    ]

    limites_x[criterio] = (
        dados_criterio["x"].min(),
        dados_criterio["x"].max()
    )


for linha, iteracao in enumerate(rounds_mostrados):

    for criterio in [1, 2]:

        ax = axes[linha, criterio - 1]

        dados = dados_recuperacao[
            (dados_recuperacao["iteration"] == iteracao) &
            (dados_recuperacao["criterion"] == criterio)
        ].copy()

        dados = dados.sort_values("x")

        # True marginal value function
        ax.plot(
            dados["x"],
            dados["real_utility"],
            linewidth=2,
            label="True"
        )

        # Estimated marginal value function after post-optimization
        ax.plot(
            dados["x"],
            dados["post_optimization"],
            marker="o",
            markersize=5,
            linewidth=1.5,
            linestyle="--",
            label="Estimated"
        )

        ax.set_xlim(limites_x[criterio])
        ax.set_ylim(0, 1)

        ax.grid(
            True,
            which="major",
            linewidth=0.6,
            alpha=0.18
        )

        ax.set_axisbelow(True)

        # Títulos apenas na primeira linha
        if linha == 0:
            ax.set_title(f"Objective {criterio}")

        # Eixo x apenas na última linha
        if linha == len(rounds_mostrados) - 1:
            ax.set_xlabel(f"Objective {criterio} value")

        # Identificação do round e eixo y na coluna esquerda
        if criterio == 1:
            ax.set_ylabel(
                f"Round {iteracao}\nMarginal value"
            )


# Legenda única
handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.995)
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)

fig.savefig(
    os.path.join(
        PASTA_FIGURAS,
        "illustrative_value_function_recovery.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nPARETO FRONT SIZES")
tamanhos_pareto = df_pareto.groupby("frontier").size()

for fronteira, n in tamanhos_pareto.items():
    print(f"P{fronteira}: {n} alternatives")

print(f"Minimum: {tamanhos_pareto.min()}")
print(f"Maximum: {tamanhos_pareto.max()}")
print(f"Total: {tamanhos_pareto.sum()}")