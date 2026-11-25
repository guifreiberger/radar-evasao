"""Previsão do risco de evasão para um novo aluno.

Uso como script (na raiz do projeto):
    python -m src.prever                                  # roda os alunos de exemplo
    python -m src.prever exemplos/aluno_risco_alto.json   # roda um arquivo específico

Uso como função:
    from src.prever import carregar_modelo, prever_evasao
    artefato = carregar_modelo()
    resultado = prever_evasao(artefato, {"curso": 9119, "idade_ingresso": 24, ...})
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.dados import CURSOS, RAIZ, adicionar_atributos

CAMINHO_MODELO = RAIZ / "models" / "modelo_evasao.joblib"
PASTA_EXEMPLOS = RAIZ / "exemplos"
# Abaixo do limiar: risco baixo. Entre o limiar e este corte: moderado. Acima: alto.
CORTE_RISCO_ALTO = 0.60
# Só mostramos fatores que mudam o risco em pelo menos 2 pontos percentuais.
REDUCAO_MINIMA = 0.02


def carregar_modelo(caminho=CAMINHO_MODELO):
    if not Path(caminho).exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {caminho}. Rode antes: python -m src.treinar"
        )
    return joblib.load(caminho)


def validar_aluno(artefato, aluno):
    """Confere campos desconhecidos e combinações impossíveis. Retorna a lista de erros."""
    erros = []
    desconhecidos = sorted(set(aluno) - set(artefato["campos_entrada"]))
    if desconhecidos:
        erros.append(f"Campos desconhecidos: {', '.join(desconhecidos)}")

    def valor(campo):
        return aluno.get(campo, artefato["valores_padrao"][campo])

    if valor("disc_1sem_aprovadas") > valor("disc_1sem_matriculadas"):
        erros.append("Disciplinas aprovadas não podem ser mais que as matriculadas.")
    if valor("disc_1sem_aprovadas") > valor("disc_1sem_avaliacoes"):
        erros.append("Disciplinas aprovadas não podem ser mais que as avaliações realizadas.")
    if not 0 <= valor("disc_1sem_nota_media") <= 20:
        erros.append("A nota média do semestre deve estar entre 0 e 20.")
    if not 0 <= valor("nota_admissao") <= 200:
        erros.append("A nota de admissão deve estar entre 0 e 200.")
    if valor("curso") not in CURSOS:
        erros.append(f"Código de curso inválido: {valor('curso')}")
    if not 15 <= valor("idade_ingresso") <= 80:
        erros.append("A idade no ingresso deve estar entre 15 e 80 anos.")
    return erros


def montar_registro(artefato, aluno):
    """Completa os campos não informados com o valor típico do treino."""
    registro = {
        campo: aluno.get(campo, padrao) for campo, padrao in artefato["valores_padrao"].items()
    }
    campos_com_padrao = sorted(set(artefato["valores_padrao"]) - set(aluno))
    return registro, campos_com_padrao


def _probabilidade(artefato, registro):
    entrada = adicionar_atributos(pd.DataFrame([registro]))[artefato["colunas_modelo"]]
    return float(artefato["pipeline"].predict_proba(entrada)[0, 1])


def classificar_risco(probabilidade, limiar):
    if probabilidade >= max(CORTE_RISCO_ALTO, limiar):
        return "alto"
    if probabilidade >= limiar:
        return "moderado"
    return "baixo"


def _cenarios_alternativos(artefato, registro):
    """Mudanças plausíveis e acionáveis para o cenário "e se...?".

    Cada item: (fator, descrição, alterações no registro).
    """
    referencia = artefato["referencia"]
    alternativas = []
    if registro["mensalidades_em_dia"] == 0:
        alternativas.append(
            ("Mensalidades", "com as mensalidades em dia", {"mensalidades_em_dia": 1})
        )
    if registro["devedor"] == 1:
        alternativas.append(
            ("Dívidas", "sem dívidas com a instituição", {"devedor": 0})
        )
    if registro["bolsista"] == 0:
        alternativas.append(("Bolsa", "com bolsa de estudos", {"bolsista": 1}))

    matriculadas = registro["disc_1sem_matriculadas"]
    aprovadas_tipicas = round(matriculadas * referencia["taxa_aprovacao_1sem"])
    if matriculadas > 0 and registro["disc_1sem_aprovadas"] < aprovadas_tipicas:
        alternativas.append(
            (
                "Aprovações no 1º semestre",
                f"com aprovação em {aprovadas_tipicas} de {int(matriculadas)} disciplinas no 1º semestre"
                " (o típico de quem não evadiu)",
                {
                    "disc_1sem_aprovadas": aprovadas_tipicas,
                    "disc_1sem_avaliacoes": max(registro["disc_1sem_avaliacoes"], aprovadas_tipicas),
                },
            )
        )
    nota_tipica = round(referencia["disc_1sem_nota_media"], 1)
    if registro["disc_1sem_nota_media"] < nota_tipica:
        alternativas.append(
            (
                "Nota média no 1º semestre",
                f"com nota média {nota_tipica:.1f} no 1º semestre (típica de quem não evadiu)".replace(".", ","),
                {"disc_1sem_nota_media": nota_tipica},
            )
        )
    return alternativas


def explicar_fatores(artefato, registro, probabilidade):
    """Para cada fator acionável, mede quanto o risco cairia se ele estivesse no
    patamar típico de alunos que não evadiram. Mostra associação aprendida pelo
    modelo, não causa."""
    fatores = []
    for fator, descricao, alteracoes in _cenarios_alternativos(artefato, registro):
        nova_probabilidade = _probabilidade(artefato, {**registro, **alteracoes})
        reducao = probabilidade - nova_probabilidade
        if reducao >= REDUCAO_MINIMA:
            fatores.append(
                {
                    "fator": fator,
                    "descricao": descricao,
                    "probabilidade_alterada": nova_probabilidade,
                    "reducao": reducao,
                }
            )
    return sorted(fatores, key=lambda item: item["reducao"], reverse=True)


def prever_evasao(artefato, aluno):
    """Recebe os dados de um aluno (dict com os campos de `campos_entrada`) e
    devolve a probabilidade de evasão, o nível de risco e os principais fatores."""
    erros = validar_aluno(artefato, aluno)
    if erros:
        raise ValueError(" ".join(erros))

    registro, campos_com_padrao = montar_registro(artefato, aluno)
    probabilidade = _probabilidade(artefato, registro)
    limiar = artefato["limiar"]
    return {
        "probabilidade_evasao": probabilidade,
        "risco": classificar_risco(probabilidade, limiar),
        "sinalizar_para_acompanhamento": probabilidade >= limiar,
        "limiar": limiar,
        "fatores": explicar_fatores(artefato, registro, probabilidade),
        "campos_com_valor_padrao": campos_com_padrao,
    }


def _imprimir_resultado(nome, aluno, resultado):
    print(f"\n=== {nome} ===")
    curso = CURSOS.get(aluno.get("curso"), "curso não informado")
    print(f"Curso: {curso} | idade: {aluno.get('idade_ingresso', '?')}")
    print(
        f"Probabilidade de evasão: {resultado['probabilidade_evasao']:.0%} "
        f"-> risco {resultado['risco'].upper()}"
    )
    if resultado["sinalizar_para_acompanhamento"]:
        print("Sugestão: incluir na lista de acompanhamento da coordenação.")
    else:
        print("Sugestão: acompanhamento de rotina.")
    if resultado["fatores"]:
        print("O que mais pesa para este aluno (segundo o modelo):")
        for item in resultado["fatores"]:
            print(
                f"  - {item['descricao'][0].upper()}{item['descricao'][1:]}, o risco cairia para "
                f"{item['probabilidade_alterada']:.0%} (-{item['reducao'] * 100:.0f} p.p.)"
            )


def ler_exemplo(arquivo):
    """Lê um JSON de aluno. Chaves iniciadas por "_" são metadados (título e
    descrição), não dados do aluno. Retorna (titulo, descricao, aluno)."""
    with open(arquivo, encoding="utf-8") as entrada:
        conteudo = json.load(entrada)
    titulo = conteudo.get("_titulo", Path(arquivo).stem)
    descricao = conteudo.get("_descricao", "")
    aluno = {campo: valor for campo, valor in conteudo.items() if not campo.startswith("_")}
    return titulo, descricao, aluno


def main(argumentos):
    sys.stdout.reconfigure(encoding="utf-8")
    artefato = carregar_modelo()
    arquivos = [Path(a) for a in argumentos] or sorted(PASTA_EXEMPLOS.glob("*.json"))
    for arquivo in arquivos:
        titulo, _, aluno = ler_exemplo(arquivo)
        _imprimir_resultado(titulo, aluno, prever_evasao(artefato, aluno))


if __name__ == "__main__":
    main(sys.argv[1:])
