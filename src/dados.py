"""Carregamento, limpeza e preparação do dataset de evasão (UCI #697).

Fonte: Realinho et al. (2021), "Predict Students' Dropout and Academic Success",
UCI Machine Learning Repository, https://doi.org/10.24432/C5MC89 (licença CC BY 4.0).
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_DADOS = RAIZ / "data" / "dados_evasao_uci.csv"

# Nome original (inglês) -> nome usado no projeto (português, snake_case)
COLUNAS = {
    "Marital status": "estado_civil",
    "Application mode": "forma_ingresso",
    "Application order": "ordem_opcao",
    "Course": "curso",
    "Daytime/evening attendance": "turno_diurno",
    "Previous qualification": "escolaridade_anterior",
    "Previous qualification (grade)": "nota_escolaridade_anterior",
    "Nacionality": "nacionalidade",
    "Mother's qualification": "escolaridade_mae",
    "Father's qualification": "escolaridade_pai",
    "Mother's occupation": "ocupacao_mae",
    "Father's occupation": "ocupacao_pai",
    "Admission grade": "nota_admissao",
    "Displaced": "deslocado",
    "Educational special needs": "necessidades_especiais",
    "Debtor": "devedor",
    "Tuition fees up to date": "mensalidades_em_dia",
    "Gender": "sexo_masculino",
    "Scholarship holder": "bolsista",
    "Age at enrollment": "idade_ingresso",
    "International": "internacional",
    "Curricular units 1st sem (credited)": "disc_1sem_creditadas",
    "Curricular units 1st sem (enrolled)": "disc_1sem_matriculadas",
    "Curricular units 1st sem (evaluations)": "disc_1sem_avaliacoes",
    "Curricular units 1st sem (approved)": "disc_1sem_aprovadas",
    "Curricular units 1st sem (grade)": "disc_1sem_nota_media",
    "Curricular units 1st sem (without evaluations)": "disc_1sem_sem_avaliacao",
    "Curricular units 2nd sem (credited)": "disc_2sem_creditadas",
    "Curricular units 2nd sem (enrolled)": "disc_2sem_matriculadas",
    "Curricular units 2nd sem (evaluations)": "disc_2sem_avaliacoes",
    "Curricular units 2nd sem (approved)": "disc_2sem_aprovadas",
    "Curricular units 2nd sem (grade)": "disc_2sem_nota_media",
    "Curricular units 2nd sem (without evaluations)": "disc_2sem_sem_avaliacao",
    "Unemployment rate": "taxa_desemprego",
    "Inflation rate": "taxa_inflacao",
    "GDP": "pib",
    "Target": "situacao",
}

SITUACOES = {"Dropout": "Evadiu", "Enrolled": "Matriculado", "Graduate": "Formado"}
ALVO = "evadiu"

CURSOS = {
    33: "Tecnologias de Produção de Biocombustíveis",
    171: "Design de Animação e Multimídia",
    8014: "Serviço Social (noturno)",
    9003: "Agronomia",
    9070: "Design de Comunicação",
    9085: "Enfermagem Veterinária",
    9119: "Engenharia Informática",
    9130: "Equinocultura",
    9147: "Gestão",
    9238: "Serviço Social",
    9254: "Turismo",
    9500: "Enfermagem",
    9556: "Higiene Oral",
    9670: "Gestão de Publicidade e Marketing",
    9773: "Jornalismo e Comunicação",
    9853: "Educação Básica",
    9991: "Gestão (noturno)",
}

FORMAS_INGRESSO = {
    1: "1ª fase - contingente geral",
    2: "Portaria 612/93",
    5: "1ª fase - contingente especial (Açores)",
    7: "Titular de outro curso superior",
    10: "Portaria 854-B/99",
    15: "Estudante internacional (licenciatura)",
    16: "1ª fase - contingente especial (Madeira)",
    17: "2ª fase - contingente geral",
    18: "3ª fase - contingente geral",
    26: "Portaria 533-A/99, alínea b2",
    27: "Portaria 533-A/99, alínea b3",
    39: "Maiores de 23 anos",
    42: "Transferência",
    43: "Mudança de curso",
    44: "Diploma de especialização tecnológica",
    51: "Mudança de instituição/curso",
    53: "Diploma de ciclo curto",
    57: "Mudança de instituição/curso (internacional)",
}

ESTADOS_CIVIS = {
    1: "Solteiro(a)",
    2: "Casado(a)",
    3: "Viúvo(a)",
    4: "Divorciado(a)",
    5: "União de facto",
    6: "Separado(a) judicialmente",
}

# ---------------------------------------------------------------------------
# Grupos de atributos
# ---------------------------------------------------------------------------
# Códigos nominais: o número não tem ordem, então viram one-hot.
CATEGORICAS = [
    "estado_civil",
    "forma_ingresso",
    "curso",
    "escolaridade_anterior",
    "nacionalidade",
    "escolaridade_mae",
    "escolaridade_pai",
    "ocupacao_mae",
    "ocupacao_pai",
]

BINARIAS = [
    "turno_diurno",
    "deslocado",
    "necessidades_especiais",
    "devedor",
    "mensalidades_em_dia",
    "sexo_masculino",
    "bolsista",
    "internacional",
]

NUMERICAS_MATRICULA = [
    "ordem_opcao",
    "nota_escolaridade_anterior",
    "nota_admissao",
    "idade_ingresso",
    "taxa_desemprego",
    "taxa_inflacao",
    "pib",
]


def _colunas_semestre(semestre):
    prefixo = f"disc_{semestre}sem"
    return [
        f"{prefixo}_creditadas",
        f"{prefixo}_matriculadas",
        f"{prefixo}_avaliacoes",
        f"{prefixo}_aprovadas",
        f"{prefixo}_nota_media",
        f"{prefixo}_sem_avaliacao",
        f"taxa_aprovacao_{semestre}sem",  # atributo criado em adicionar_atributos()
    ]


SEMESTRE_1 = _colunas_semestre(1)
SEMESTRE_2 = _colunas_semestre(2)

# Cada cenário corresponde a um momento em que a coordenação faria a previsão.
# Quanto mais tarde, mais informação (e melhor o modelo), mas menos tempo para agir.
CENARIOS = {
    "matricula": {
        "titulo": "Na matrícula",
        "descricao": "Só dados conhecidos no ingresso (perfil, histórico e situação financeira).",
        "numericas": NUMERICAS_MATRICULA,
    },
    "1_semestre": {
        "titulo": "Fim do 1º semestre",
        "descricao": "Dados da matrícula + desempenho no 1º semestre.",
        "numericas": NUMERICAS_MATRICULA + SEMESTRE_1,
    },
    "2_semestre": {
        "titulo": "Fim do 2º semestre",
        "descricao": "Dados da matrícula + desempenho no 1º e 2º semestres.",
        "numericas": NUMERICAS_MATRICULA + SEMESTRE_1 + SEMESTRE_2,
    },
}


def atributos_do_cenario(nome_cenario):
    """Retorna (numericas, binarias, categoricas) usadas no cenário."""
    cenario = CENARIOS[nome_cenario]
    return list(cenario["numericas"]), list(BINARIAS), list(CATEGORICAS)


def todas_colunas_do_cenario(nome_cenario):
    numericas, binarias, categoricas = atributos_do_cenario(nome_cenario)
    return numericas + binarias + categoricas


# ---------------------------------------------------------------------------
# Carregamento e preparação
# ---------------------------------------------------------------------------
def carregar_dados_brutos(caminho=CAMINHO_DADOS):
    """Lê o CSV original (separador ';') e padroniza os nomes das colunas."""
    df = pd.read_csv(caminho, sep=";", encoding="utf-8-sig")
    # O cabeçalho original tem um TAB em "Daytime/evening attendance\t".
    df.columns = [coluna.strip() for coluna in df.columns]
    colunas_faltando = set(COLUNAS) - set(df.columns)
    if colunas_faltando:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(colunas_faltando)}")
    return df.rename(columns=COLUNAS)


def adicionar_atributos(df):
    """Cria atributos derivados. Usado no treino e na previsão (mesma regra nos dois)."""
    df = df.copy()
    for semestre in (1, 2):
        matriculadas = f"disc_{semestre}sem_matriculadas"
        aprovadas = f"disc_{semestre}sem_aprovadas"
        if matriculadas in df.columns and aprovadas in df.columns:
            # Aluno sem disciplinas matriculadas no semestre recebe taxa 0.
            df[f"taxa_aprovacao_{semestre}sem"] = np.where(
                df[matriculadas] > 0, df[aprovadas] / df[matriculadas].clip(lower=1), 0.0
            )
    return df


def carregar_dados(caminho=CAMINHO_DADOS):
    """Dataset pronto para análise: nomes em português, situação traduzida,
    alvo binário `evadiu` (1 = evadiu, 0 = formado ou ainda matriculado)."""
    df = carregar_dados_brutos(caminho)
    df["situacao"] = df["situacao"].map(SITUACOES)
    df[ALVO] = (df["situacao"] == "Evadiu").astype(int)
    df["nome_curso"] = df["curso"].map(CURSOS)
    return adicionar_atributos(df)
