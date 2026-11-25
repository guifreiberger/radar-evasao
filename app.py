"""Radar de Evasão: demonstração interativa do modelo.

Uso (na raiz do projeto):
    streamlit run app.py
"""

import streamlit as st

from src.dados import CURSOS, ESTADOS_CIVIS, FORMAS_INGRESSO
from src.prever import (
    CORTE_RISCO_ALTO,
    PASTA_EXEMPLOS,
    carregar_modelo,
    ler_exemplo,
    prever_evasao,
)

st.set_page_config(page_title="Radar de Evasão", page_icon="🎓", layout="wide")


@st.cache_resource
def obter_modelo():
    return carregar_modelo()


@st.cache_data
def obter_exemplos():
    """{título: (descrição, aluno)} dos alunos fictícios da pasta exemplos/."""
    exemplos = {}
    for arquivo in sorted(PASTA_EXEMPLOS.glob("*.json")):
        titulo, descricao, aluno = ler_exemplo(arquivo)
        exemplos[titulo] = (descricao, aluno)
    return exemplos


def porcentagem(valor):
    return f"{valor:.0%}"


def decimal(valor, casas=2):
    return f"{valor:.{casas}f}".replace(".", ",")


artefato = obter_modelo()
limiar = artefato["limiar"]
metricas = artefato["metricas_teste"]

# ---------------------------------------------------------------------------
# Barra lateral: sobre o modelo
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Sobre o modelo")
    st.markdown(
        f"""
**{artefato['nome_modelo']}**, momento da previsão: **{artefato['titulo_cenario'].lower()}**.

Treinado com 3.539 alunos de graduação (UCI, *Predict Students' Dropout and Academic Success*).

**No teste (885 alunos nunca vistos):**
- identificou **{porcentagem(metricas['recall'])}** dos alunos que evadiram (recall)
- **{porcentagem(metricas['precisao'])}** dos sinalizados evadiram de fato (precisão)
- ROC-AUC **{decimal(metricas['roc_auc'])}** · PR-AUC **{decimal(metricas['pr_auc'])}**

**Níveis de risco**
- 🟢 **Baixo:** abaixo de {porcentagem(limiar)}
- 🟡 **Moderado:** de {porcentagem(limiar)} a {porcentagem(CORTE_RISCO_ALTO)}
- 🔴 **Alto:** {porcentagem(CORTE_RISCO_ALTO)} ou mais
"""
    )
    st.caption(
        "Ferramenta de apoio à decisão. O resultado indica alunos que podem precisar "
        "de acompanhamento. Não deve ser usado para punir ou restringir o acesso de ninguém."
    )

# ---------------------------------------------------------------------------
# Entrada de dados
# ---------------------------------------------------------------------------
st.title("🎓 Radar de Evasão")
st.markdown(
    "Estima a **probabilidade de um aluno de graduação evadir** a partir dos dados da matrícula "
    "e do desempenho no **1º semestre**, e mostra o que mais pesa para aquele aluno."
)

exemplos = obter_exemplos()
opcao = st.selectbox(
    "Começar a partir de:",
    ["Aluno típico (valores mais comuns na base)"] + list(exemplos),
)
descricao_exemplo, aluno_exemplo = exemplos.get(opcao, ("", {}))
if descricao_exemplo:
    st.caption(descricao_exemplo)
aluno_inicial = {**artefato["valores_padrao"], **aluno_exemplo}

with st.form("dados_aluno"):
    coluna_matricula, coluna_semestre = st.columns(2, gap="large")

    with coluna_matricula:
        st.subheader("Dados da matrícula")
        codigos_cursos = sorted(CURSOS, key=CURSOS.get)
        curso = st.selectbox(
            "Curso",
            codigos_cursos,
            index=codigos_cursos.index(int(aluno_inicial["curso"])),
            format_func=CURSOS.get,
        )
        turno_diurno = st.radio(
            "Turno",
            [1, 0],
            index=[1, 0].index(int(aluno_inicial["turno_diurno"])),
            format_func={1: "Diurno", 0: "Noturno"}.get,
            horizontal=True,
        )
        codigos_ingresso = list(FORMAS_INGRESSO)
        forma_ingresso = st.selectbox(
            "Forma de ingresso",
            codigos_ingresso,
            index=codigos_ingresso.index(int(aluno_inicial["forma_ingresso"])),
            format_func=FORMAS_INGRESSO.get,
        )
        idade_ingresso = st.number_input(
            "Idade no ingresso", min_value=17, max_value=70, value=int(aluno_inicial["idade_ingresso"])
        )
        sexo_masculino = st.radio(
            "Sexo",
            [0, 1],
            index=[0, 1].index(int(aluno_inicial["sexo_masculino"])),
            format_func={0: "Feminino", 1: "Masculino"}.get,
            horizontal=True,
        )
        codigos_estado_civil = list(ESTADOS_CIVIS)
        estado_civil = st.selectbox(
            "Estado civil",
            codigos_estado_civil,
            index=codigos_estado_civil.index(int(aluno_inicial["estado_civil"])),
            format_func=ESTADOS_CIVIS.get,
        )
        nota_admissao = st.number_input(
            "Nota de admissão (0 a 200)",
            min_value=0.0,
            max_value=200.0,
            value=float(aluno_inicial["nota_admissao"]),
            step=0.5,
        )
        deslocado = st.checkbox(
            "Mora fora da cidade de origem", value=bool(aluno_inicial["deslocado"])
        )
        bolsista = st.checkbox("Bolsista", value=bool(aluno_inicial["bolsista"]))
        mensalidades_em_dia = st.checkbox(
            "Mensalidades em dia", value=bool(aluno_inicial["mensalidades_em_dia"])
        )
        devedor = st.checkbox(
            "Tem dívidas com a instituição", value=bool(aluno_inicial["devedor"])
        )

    with coluna_semestre:
        st.subheader("Desempenho no 1º semestre")
        disc_1sem_matriculadas = st.number_input(
            "Disciplinas matriculadas", 0, 26, int(aluno_inicial["disc_1sem_matriculadas"])
        )
        disc_1sem_avaliacoes = st.number_input(
            "Avaliações realizadas (provas e trabalhos)",
            0,
            45,
            int(aluno_inicial["disc_1sem_avaliacoes"]),
        )
        disc_1sem_aprovadas = st.number_input(
            "Disciplinas aprovadas", 0, 26, int(aluno_inicial["disc_1sem_aprovadas"])
        )
        disc_1sem_nota_media = st.number_input(
            "Nota média nas disciplinas (0 a 20)",
            min_value=0.0,
            max_value=20.0,
            value=float(aluno_inicial["disc_1sem_nota_media"]),
            step=0.1,
        )
        disc_1sem_creditadas = st.number_input(
            "Disciplinas aproveitadas de outro curso",
            0,
            20,
            int(aluno_inicial["disc_1sem_creditadas"]),
        )
        disc_1sem_sem_avaliacao = st.number_input(
            "Disciplinas sem nenhuma avaliação",
            0,
            12,
            int(aluno_inicial["disc_1sem_sem_avaliacao"]),
        )
        st.caption(
            "Na base original (Portugal) as notas vão de 0 a 20, e a aprovação é a partir de 10."
        )

    enviar = st.form_submit_button("Calcular risco", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------
if enviar:
    aluno = {
        "curso": curso,
        "turno_diurno": turno_diurno,
        "forma_ingresso": forma_ingresso,
        "idade_ingresso": idade_ingresso,
        "sexo_masculino": sexo_masculino,
        "estado_civil": estado_civil,
        "nota_admissao": nota_admissao,
        "deslocado": int(deslocado),
        "bolsista": int(bolsista),
        "mensalidades_em_dia": int(mensalidades_em_dia),
        "devedor": int(devedor),
        "disc_1sem_matriculadas": disc_1sem_matriculadas,
        "disc_1sem_avaliacoes": disc_1sem_avaliacoes,
        "disc_1sem_aprovadas": disc_1sem_aprovadas,
        "disc_1sem_nota_media": disc_1sem_nota_media,
        "disc_1sem_creditadas": disc_1sem_creditadas,
        "disc_1sem_sem_avaliacao": disc_1sem_sem_avaliacao,
    }
    try:
        resultado = prever_evasao(artefato, aluno)
    except ValueError as erro:
        st.error(f"Verifique os dados: {erro}")
        st.stop()

    probabilidade = resultado["probabilidade_evasao"]
    st.divider()
    coluna_numero, coluna_texto = st.columns([1, 2], gap="large")
    with coluna_numero:
        st.metric("Probabilidade de evasão", porcentagem(probabilidade))
        st.progress(min(max(probabilidade, 0.0), 1.0))
    with coluna_texto:
        if resultado["risco"] == "alto":
            st.error(
                "🔴 **Risco alto.** Prioridade para atendimento individual com a coordenação."
            )
        elif resultado["risco"] == "moderado":
            st.warning(
                "🟡 **Risco moderado.** Incluir no acompanhamento: contato do tutor, "
                "convite para monitoria."
            )
        else:
            st.success("🟢 **Risco baixo.** Acompanhamento de rotina.")
        st.caption(
            f"De cada 10 alunos que recebem essa probabilidade, cerca de {round(probabilidade * 10)} "
            "evadem: no teste, as probabilidades do modelo se mostraram bem calibradas."
        )

    st.subheader("O que mais pesa para este aluno")
    if resultado["fatores"]:
        for fator in resultado["fatores"]:
            st.markdown(
                f"- **{fator['descricao'][0].upper()}{fator['descricao'][1:]}**, o risco cairia "
                f"de {porcentagem(probabilidade)} para **{porcentagem(fator['probabilidade_alterada'])}** "
                f"(−{fator['reducao'] * 100:.0f} pontos percentuais)."
            )
        st.caption(
            "Cada linha simula uma mudança isolada, mantendo o resto igual. É o que o modelo "
            "aprendeu dos dados (associação), não uma garantia de causa e efeito."
        )
    else:
        st.markdown(
            "Nenhum fator acionável reduziria o risco de forma relevante: este aluno já está "
            "no patamar típico de quem não evade."
        )

    with st.expander("Campos preenchidos automaticamente"):
        st.write(
            "Estes dados não aparecem no formulário e foram preenchidos com o valor mais comum "
            "(ou a mediana) dos alunos de treino:"
        )
        st.write(", ".join(resultado["campos_com_valor_padrao"]))
