"""Construção, comparação, ajuste e avaliação dos modelos de evasão."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.dados import ALVO, CENARIOS, atributos_do_cenario, todas_colunas_do_cenario

SEMENTE = 42
PROPORCAO_TESTE = 0.20
N_FOLDS = 5
# Categorias com menos de 20 alunos no treino são agrupadas em "infrequente".
MIN_FREQUENCIA_CATEGORIA = 20
# F2: dá o dobro de peso ao recall. Deixar de identificar um aluno em risco custa
# mais do que conversar com um aluno que não iria evadir.
BETA = 2

METRICAS_CV = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": make_scorer(f1_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "precisao": make_scorer(precision_score, zero_division=0),
}

NOMES_MODELOS = [
    "Baseline (proporção)",
    "Regressão Logística",
    "Random Forest",
    "Gradient Boosting",
]

# Modelos que precisam de atributos padronizados (sensíveis à escala).
PRECISA_ESCALA = {"Regressão Logística"}

GRADES = {
    "Regressão Logística": {"modelo__C": [0.01, 0.1, 1.0, 10.0]},
    "Random Forest": {
        "modelo__min_samples_leaf": [1, 3, 5],
        "modelo__max_features": ["sqrt", 0.5],
    },
    "Gradient Boosting": {
        "modelo__learning_rate": [0.03, 0.1],
        "modelo__max_leaf_nodes": [7, 15, 31],
        "modelo__max_iter": [200, 400],
    },
}


def criar_estimador(nome_modelo):
    """Instancia o classificador. Nenhum usa class_weight: o desbalanceamento
    (32% evadiu) é tratado escolhendo o limiar de decisão, o que mantém as
    probabilidades interpretáveis."""
    if nome_modelo == "Baseline (proporção)":
        return DummyClassifier(strategy="prior")
    if nome_modelo == "Regressão Logística":
        return LogisticRegression(C=1.0, max_iter=5000)
    if nome_modelo == "Random Forest":
        return RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=SEMENTE
        )
    if nome_modelo == "Gradient Boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=SEMENTE,
        )
    raise ValueError(f"Modelo desconhecido: {nome_modelo}")


def criar_pipeline(nome_modelo, nome_cenario, excluir=()):
    """Pré-processamento + modelo num único Pipeline.

    O pré-processamento é ajustado dentro de cada fold da validação cruzada,
    evitando vazamento de informação do conjunto de validação para o treino.
    `excluir` remove atributos (usado nos testes de sensibilidade).
    """
    numericas, binarias, categoricas = (
        [coluna for coluna in grupo if coluna not in excluir]
        for grupo in atributos_do_cenario(nome_cenario)
    )
    escalonador = StandardScaler() if nome_modelo in PRECISA_ESCALA else "passthrough"
    preprocessador = ColumnTransformer(
        transformers=[
            ("numericas", escalonador, numericas),
            ("binarias", "passthrough", binarias),
            (
                "categoricas",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_FREQUENCIA_CATEGORIA,
                    sparse_output=False,
                ),
                categoricas,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessamento", preprocessador), ("modelo", criar_estimador(nome_modelo))])


def dividir_dados(df):
    """Separa 20% para teste final, estratificado pela situação (3 classes),
    para manter a mesma proporção de evadidos, matriculados e formados."""
    treino, teste = train_test_split(
        df, test_size=PROPORCAO_TESTE, stratify=df["situacao"], random_state=SEMENTE
    )
    return treino.reset_index(drop=True), teste.reset_index(drop=True)


def validacao_cruzada():
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMENTE)


def comparar_modelos(treino, nomes_cenarios=None, nomes_modelos=None):
    """Validação cruzada (5 folds) de cada modelo em cada cenário, só no treino."""
    nomes_cenarios = nomes_cenarios or list(CENARIOS)
    nomes_modelos = nomes_modelos or NOMES_MODELOS
    linhas = []
    for nome_cenario in nomes_cenarios:
        X = treino[todas_colunas_do_cenario(nome_cenario)]
        y = treino[ALVO]
        for nome_modelo in nomes_modelos:
            resultado = cross_validate(
                criar_pipeline(nome_modelo, nome_cenario),
                X,
                y,
                cv=validacao_cruzada(),
                scoring=METRICAS_CV,
                n_jobs=-1,
            )
            linha = {"cenario": nome_cenario, "modelo": nome_modelo}
            for metrica in METRICAS_CV:
                valores = resultado[f"test_{metrica}"]
                linha[metrica] = valores.mean()
                linha[f"{metrica}_dp"] = valores.std()
            linhas.append(linha)
    return pd.DataFrame(linhas)


def ajustar_hiperparametros(nome_modelo, nome_cenario, treino):
    """Busca em grade (validação cruzada de 5 folds), otimizando a PR-AUC."""
    busca = GridSearchCV(
        criar_pipeline(nome_modelo, nome_cenario),
        param_grid=GRADES[nome_modelo],
        scoring="average_precision",
        cv=validacao_cruzada(),
        n_jobs=-1,
    )
    busca.fit(treino[todas_colunas_do_cenario(nome_cenario)], treino[ALVO])
    return busca


def probabilidades_fora_da_amostra(pipeline, nome_cenario, treino):
    """Probabilidade de cada aluno do treino prevista por um modelo que NÃO o viu
    (cross_val_predict). É com elas que escolhemos o limiar, sem tocar no teste."""
    return cross_val_predict(
        pipeline,
        treino[todas_colunas_do_cenario(nome_cenario)],
        treino[ALVO],
        cv=validacao_cruzada(),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]


def escolher_limiar(y_verdadeiro, probabilidades, beta=BETA):
    """Limiar que maximiza o F-beta."""
    limiares = np.round(np.arange(0.05, 0.951, 0.01), 2)
    pontuacoes = [
        fbeta_score(y_verdadeiro, probabilidades >= limiar, beta=beta, zero_division=0)
        for limiar in limiares
    ]
    melhor = int(np.argmax(pontuacoes))
    return float(limiares[melhor]), pd.DataFrame({"limiar": limiares, f"f{beta}": pontuacoes})


def calcular_metricas(y_verdadeiro, probabilidades, limiar):
    previsto = (probabilidades >= limiar).astype(int)
    vn, fp, fn, vp = confusion_matrix(y_verdadeiro, previsto, labels=[0, 1]).ravel()
    return {
        "limiar": float(limiar),
        "roc_auc": float(roc_auc_score(y_verdadeiro, probabilidades)),
        "pr_auc": float(average_precision_score(y_verdadeiro, probabilidades)),
        "acuracia": float(accuracy_score(y_verdadeiro, previsto)),
        "precisao": float(precision_score(y_verdadeiro, previsto, zero_division=0)),
        "recall": float(recall_score(y_verdadeiro, previsto)),
        "f1": float(f1_score(y_verdadeiro, previsto)),
        f"f{BETA}": float(fbeta_score(y_verdadeiro, previsto, beta=BETA)),
        "verdadeiros_positivos": int(vp),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdadeiros_negativos": int(vn),
    }
