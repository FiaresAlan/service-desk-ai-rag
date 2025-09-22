import langchain_google_genai
import langchain
import toml
from pydantic import BaseModel, Field
from typing import Literal, List, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from IPython.display import display, Image

# ============================================================
# 🔑 Carregando chave da API
# ============================================================
secrets = toml.load("secrets.toml")
api_key_gemini = secrets["default"]["GEMINI_API_KEY"]

docs = []

# ============================================================
# ⚙️ Configuração inicial do LLM (teste simples de pergunta)
# ============================================================
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0,
    api_key=api_key_gemini
    # verbose=True  # habilitar se quiser mais logs
)

pergunta = input("faça uma pergunta a IA: ")
resposta_teste = llm.invoke(pergunta)
print(resposta_teste.content)

# ============================================================
# 📌 Prompt de triagem
# ============================================================
TRIAGEM_PROMPT = (
    "Você é um triador de Service Desk para políticas internas da empresa Carraro Desenvolvimento. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
    "{\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
    '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n"
    "Regras:\n"
    '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas.\n'
    '- **PEDIR_INFO**: Mensagens vagas ou sem contexto suficiente.\n'
    '- **ABRIR_CHAMADO**: Pedidos de exceção, aprovação ou abertura explícita de chamado.\n'
)

# ============================================================
# 📦 Modelo de saída estruturada para triagem
# ============================================================
class TriagemOut(BaseModel):
  decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
  urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
  campos_faltantes: List[str] = Field(default_factory=list)

# ============================================================
# ⚙️ Configuração do LLM para triagem estruturada
# ============================================================
llm_triagem = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=1.0,
    api_key=api_key_gemini,
)

# Chain que garante resposta no formato TriagemOut
triagem_chain = llm_triagem.with_structured_output(TriagemOut)

def triagem(mensagem_humana: str) -> Dict:
    """
    Executa a triagem de uma mensagem.
    Retorna um dicionário estruturado com decisão, urgência e campos faltantes.
    """
    saida: TriagemOut = triagem_chain.invoke([
       SystemMessage(content=TRIAGEM_PROMPT),
       HumanMessage(content=mensagem_humana)
    ]) 
    return saida.model_dump()

# ============================================================
# 🔍 Testes iniciais da triagem
# ============================================================
testes = [
    "Posso reembolsar a internet?",
    "Quero mais 5 dias de trabalho remoto, como faço?",
    "Quantas capivaras tem no Rio Pinheiros?"
]
for msg_teste in testes:
   print(f"Pergunta: {msg_teste} \n > Resposta: {triagem(msg_teste)}\n")

# ============================================================
# 📂 Carregamento de PDFs
# ============================================================
for n in Path("pdf-agent-development").glob("*.pdf"):
    try:
        loader = PyMuPDFLoader(str(n))
        docs.extend(loader.load())
        print(f"Arquivo carregado com sucesso {n.name}.")
    except Exception as e:
        print(f"Erro ao carregar o arquivo {n.name}: {e}.")
print(f"Total de documentos carregados: {len(docs)}.")

# ============================================================
# ✂️ Split dos documentos (chunking com overlap)
# ============================================================
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(docs)

# ============================================================
# 🔢 Embeddings com Gemini
# ============================================================
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key_gemini 
)

# ============================================================
# 🗄️ Vetorizar documentos com FAISS
# ============================================================
vectorstores = FAISS.from_documents(chunks, embeddings)

# ============================================================
# 🔎 Configuração do Retriever
# ============================================================
retriever = vectorstores.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={"score_threshold":0.3, "k":4}
)

# ============================================================
# 📝 Prompt para RAG
# ============================================================
prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
    "Você é um assistente de políticas internas (RH/IT) da empresa FiaresDev. "
    "Responda SOMENTE com base no contexto fornecido. "
    "Se não houver base suficiente, responda apenas 'Não sei'."),
    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])

# ============================================================
# 🔗 Cadeia para combinar documentos e responder
# ============================================================
document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)

def perguntar_politica_RAG(pergunta: str) -> Dict:
    """
    Faz uma pergunta ao RAG e retorna resposta + contexto encontrado.
    """
    docs_relacionados = retriever.invoke(pergunta)
    if not docs_relacionados:
        return {"answer": "Não sei.", "citacoes": [], "contexto_encontrado": False}
 
    answer = document_chain.invoke({"input": pergunta, "context": docs_relacionados})
    txt = (answer or "").strip()

    if txt.rstrip(".!?") == "Não sei":
        return {"answer": "Não sei.", "citacoes": [], "contexto_encontrado": False}
    
    return {"answer": txt, "citacoes": docs_relacionados, "contexto_encontrado": True}

# ============================================================
# 🔍 Testes do RAG
# ============================================================
testes = [
    "Posso reembolsar a internet?",
    "Quero mais 5 dias de trabalho remoto, como faço?",
    "Quantas capivaras tem no Rio Pinheiros?"
]

for msg_teste in testes:
  resposta = perguntar_politica_RAG(msg_teste)
  print(f"Pergunta: {msg_teste}")
  print(f"Resposta: {resposta['answer']}")
  if resposta['contexto_encontrado']:
      print("CITAÇÕES:")
      print(resposta['citacoes'])
      print("--------------------------\n")

# ============================================================
# 🧩 Definição do estado do agente
# ============================================================
class AgentState(TypedDict, total = False):
  pergunta: str
  triagem: dict
  resposta: Optional[str]
  citacoes: List[dict]
  rag_sucesso: bool
  acao_final: str

# ============================================================
# 🔗 Nós do LangGraph
# ============================================================
def node_triagem(state: AgentState) -> AgentState:
    print("Executando nó de triagem...")
    return {
        **state,  # mantém todas as chaves já existentes
        "triagem": triagem(state["pergunta"])
    }
def node_auto_resolver(state: AgentState) -> AgentState:
    print("Executando nó de auto_resolver...")
    resposta_rag = perguntar_politica_RAG(state["pergunta"])

    update: AgentState = {
        "resposta": resposta_rag["answer"],
        "citacoes": resposta_rag.get("citacoes", []),
        "rag_sucesso": resposta_rag["contexto_encontrado"],
    }

    if resposta_rag["contexto_encontrado"]:
        update["acao_final"] = "AUTO_RESOLVER"

    return update

def node_pedir_info(state: AgentState) -> AgentState:
    print("Executando nó de pedir-info...")
    faltantes = state["triagem"].get("campos_faltantes", [])
    detalhe = ",".join(faltantes) if faltantes else "Tema e contexto específico"
    return {
        "resposta": f"Para avançar preciso que detalhe: {detalhe}",
        "citacoes": [],
        "acao_final": "PEDIR_INFO"}

def node_abrir_chamado(state: AgentState) -> AgentState:
    print("Executando nó de abrir-chamado...")
    triagem = state["triagem"]

    # Poderia ser conectado a ferramentas externas (ex.: Jira, ServiceNow)
    return {
        "resposta": f"Abrindo chamado com urgência {triagem['urgencia']}. Descrição: {state['pergunta'][:140]}",
        "citacoes": [],
        "acao_final": "ABRIR_CHAMADO"
    }

# ============================================================
# 🔑 Regras de decisão pós-triagem
# ============================================================
KEYWORDS_ABRIR_TICKET = ["aprovação", "exceção", "liberação", "abrir ticket", "abrir chamado", "acesso especial"]

def decidir_pos_triagem(state: AgentState) -> str:
    print("Decidindo após a triagem...")
    decisao = state["triagem"]["decisao"]

    if decisao == "AUTO_RESOLVER": return "resolver-auto"
    if decisao == "PEDIR_INFO": return "info-pedir"
    if decisao == "ABRIR_CHAMADO": return "chamado-abrir"

def decidir_pos_auto_resolver(state: AgentState) -> str:
    print("Decidindo após o auto-resolver...")
    if state.get("rag_sucesso"):
        print("RAG com sucesso, finalizando o fluxo.")
        return "OK"

    state_da_pergunta = (state["pergunta"] or "").lower()

    if any(k in state_da_pergunta for k in KEYWORDS_ABRIR_TICKET):
        print("RAG falhou, mas foram encontradas keywords de abertura de ticket, abrindo...")
        return "chamado-abrir"

    return "pedir-info"

# ============================================================
# 🔄 Construção do workflow com LangGraph
# ============================================================
workflow = StateGraph(AgentState)

workflow.add_node("triagem", node_triagem)
workflow.add_node("auto-resolver", node_auto_resolver)
workflow.add_node("pedir-info", node_pedir_info)
workflow.add_node("abrir-chamado", node_abrir_chamado)

workflow.add_edge(START, "triagem")

workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
    "resolver-auto": "auto-resolver",
    "info-pedir": "pedir-info",
    "chamado-abrir": "abrir-chamado"
})

workflow.add_conditional_edges("auto-resolver", decidir_pos_auto_resolver, {
    "info": "pedir-info",
    "chamado-abrir": "abrir-chamado",
    "OK": END
})

workflow.add_edge("pedir-info", END)
workflow.add_edge("abrir-chamado", END)

grafo = workflow.compile()

# ============================================================
# 📊 Renderização do grafo de fluxo
# ============================================================
graph_bytes = grafo.get_graph().draw_mermaid_png()
display(Image(graph_bytes))

testes = ["Posso reembolsar a internet?",
          "Quero mais 5 dias de trabalho remoto. Como faço?",
          "Posso reembolsar cursos ou treinamentos da Alura?",
          "É possível reembolsar certificações do Google Cloud?",
          "Posso obter o Google Gemini de graça?",
          "Qual é a palavra-chave da aula de hoje?",
          "Quantas capivaras tem no Rio Pinheiros?"]

for msg_test in testes:
    resposta_final = grafo.invoke({"pergunta": msg_test})

    triag = resposta_final.get("triagem", {})
    print(f"PERGUNTA: {msg_test}")
    print(f"DECISÃO: {triag.get('decisao')} | URGÊNCIA: {triag.get('urgencia')} | AÇÃO FINAL: {resposta_final.get('acao_final')}")
    print(f"RESPOSTA: {resposta_final.get('resposta')}")

    citacoes = resposta_final.get("citacoes", [])
    if citacoes:
        print("CITAÇÕES:")
        for c in citacoes:
            # se c for Document, acessar metadata
            if hasattr(c, "metadata"):
                doc_name = c.metadata.get("source", "desconhecido")
                pagina = c.metadata.get("page", 0) + 1
                trecho = c.page_content[:200]  # mantém lógica de exibir trecho
            else:
                # se já for dict (como vindo de formatar_citacoes)
                doc_name = c.get("documento", "desconhecido")
                pagina = c.get("pagina", 0)
                trecho = c.get("trecho", "")

            print(f" - Documento: {doc_name}, Página: {pagina}")
            print(f"   Trecho: {trecho}")

    print("------------------------------------")
