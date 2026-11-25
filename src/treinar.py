"""Treina o modelo final e salva o artefato usado pela demonstração.

Uso (na raiz do projeto):
    python -m src.treinar
"""

import json
import sys
from datetime import date

import joblib
import sklearn

from src.dados import (
    ALVO,
    CENARIOS,
    RAIZ,
    atributos_do_cenario,
    carregar_dados,
    todas_colunas_do_cenario,
)
from src.modelo import (
    BETA,
    calcular_metricas,
    comparar_modelos,
    dividir_dados,
    ajustar_hiperparametros,
    escolher_limiar,
    probabilidades_fora_da_amostra,
)

CENARIO_FINAL = "1_semestre"
CAMINHO_MODELO = RAIZ / "models" / "modelo_evasao.joblib"
PASTA_RELATORIOS = RAIZ / "reports"


def campos_de_entrada(nome_cenario):
    """Campos que o usuário informa. Atributos derivados (taxa_aprovacao_*)
    são calculados automaticamente em adicionar_atributos()."""
    return [c for c in todas_colunas_do_cenario(nome_cenario) if not c.startswith("taxa_aprovacao")]


def valores_padrao(treino, nome_cenario):
    """Valor usado quando um campo não é informado: mediana (numéricos) ou moda
    (binários e categóricos)."""
    numericas, _, _ = atributos_do_cenario(nome_cenario)
    padroes = {}
    for campo in campos_de_entrada(nome_cenario):
        if campo in numericas:
            padroes[campo] = float(treino[campo].median())
        else:
            padroes[campo] = int(treino[campo].mode().iloc[0])
    return padroes


def perfil_referencia(treino):
    """Desempenho típico de quem NÃO evadiu, usado nas explicações do tipo
    "e se este aluno tivesse o desempenho de um aluno típico?"."""
    nao_evadiu = treino[treino[ALVO] == 0]
    com_nota = nao_evadiu[nao_evadiu["disc_1sem_nota_media"] > 0]
    return {
        "taxa_aprovacao_1sem": float(nao_evadiu["taxa_aprovacao_1sem"].median()),
        "disc_1sem_nota_media": float(com_nota["disc_1sem_nota_media"].median()),
    }


def treinar_modelo_final(treino, teste, nome_cenario, nome_modelo):
    busca = ajustar_hiperparametros(nome_modelo, nome_cenario, treino)
    pipeline = busca.best_estimator_  # já reajustado em todo o conjunto de treino

    prob_oof = probabilidades_fora_da_amostra(pipeline, nome_cenario, treino)
    limiar, curva_limiar = escolher_limiar(treino[ALVO], prob_oof)

    X_teste = teste[todas_colunas_do_cenario(nome_cenario)]
    prob_teste = pipeline.predict_proba(X_teste)[:, 1]

    return {
        "pipeline": pipeline,
        "melhores_parametros": {k.replace("modelo__", ""): v for k, v in busca.best_params_.items()},
        "pr_auc_cv": float(busca.best_score_),
        "limiar": limiar,
        "curva_limiar": curva_limiar,
        "prob_oof": prob_oof,
        "prob_teste": prob_teste,
        "metricas_validacao": calcular_metricas(treino[ALVO], prob_oof, limiar),
        "metricas_teste": calcular_metricas(teste[ALVO], prob_teste, limiar),
    }


def montar_artefato(resultado, treino, nome_cenario, nome_modelo):
    return {
        "pipeline": resultado["pipeline"],
        "nome_modelo": nome_modelo,
        "cenario": nome_cenario,
        "titulo_cenario": CENARIOS[nome_cenario]["titulo"],
        "colunas_modelo": todas_colunas_do_cenario(nome_cenario),
        "campos_entrada": campos_de_entrada(nome_cenario),
        "valores_padrao": valores_padrao(treino, nome_cenario),
        "referencia": perfil_referencia(treino),
        "limiar": resultado["limiar"],
        "beta": BETA,
        "melhores_parametros": resultado["melhores_parametros"],
        "metricas_validacao": resultado["metricas_validacao"],
        "metricas_teste": resultado["metricas_teste"],
        "taxa_evasao_treino": float(treino[ALVO].mean()),
        "versao_sklearn": sklearn.__version__,
        "data_treino": date.today().isoformat(),
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df = carregar_dados()
    treino, teste = dividir_dados(df)
    print(f"Treino: {len(treino)} alunos | Teste: {len(teste)} alunos")

    print("Comparando modelos (validação cruzada, 5 folds)...")
    comparacao = comparar_modelos(treino)
    PASTA_RELATORIOS.mkdir(exist_ok=True)
    comparacao.round(4).to_csv(PASTA_RELATORIOS / "comparacao_modelos.csv", index=False)
    print(comparacao[["cenario", "modelo", "roc_auc", "pr_auc", "f1"]].round(3).to_string(index=False))

    candidatos = comparacao[
        (comparacao["cenario"] == CENARIO_FINAL) & (~comparacao["modelo"].str.startswith("Baseline"))
    ]
    nome_modelo = candidatos.sort_values("pr_auc", ascending=False)["modelo"].iloc[0]
    print(f"\nMelhor modelo no cenário '{CENARIO_FINAL}': {nome_modelo}. Ajustando hiperparâmetros...")

    resultado = treinar_modelo_final(treino, teste, CENARIO_FINAL, nome_modelo)
    artefato = montar_artefato(resultado, treino, CENARIO_FINAL, nome_modelo)

    CAMINHO_MODELO.parent.mkdir(exist_ok=True)
    joblib.dump(artefato, CAMINHO_MODELO)

    resumo = {
        "cenario": CENARIO_FINAL,
        "modelo": nome_modelo,
        "melhores_parametros": artefato["melhores_parametros"],
        "limiar": artefato["limiar"],
        "metricas_validacao_cruzada": artefato["metricas_validacao"],
        "metricas_teste": artefato["metricas_teste"],
    }
    with open(PASTA_RELATORIOS / "metricas_modelo_final.json", "w", encoding="utf-8") as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2, default=str)

    print(f"Parâmetros: {artefato['melhores_parametros']} | limiar: {artefato['limiar']:.2f}")
    print("Métricas no teste:")
    for nome, valor in artefato["metricas_teste"].items():
        print(f"  {nome}: {valor:.3f}" if isinstance(valor, float) else f"  {nome}: {valor}")
    print(f"\nModelo salvo em {CAMINHO_MODELO.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
