import langchain_google_genai
import langchain
import toml
from pydantic import BaseModel, Field
from typing import Literal, List, Dict
#Literal = Pq no prompt tem a opcao de resolver, pedindo informação
#ou abrindo chamado (3 literais). Depois temos o List =  a saida
# vai ter alguns campos: Decisao, urgencia ou campos_faltantes.
# e Dict=todo o dicionario que tem la, por ex. decisao= autoresolver | pedir_info | abrir_chamado
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader

#pymupdf: Lê pdf | langchain_community: Ajuda conectar tudo  |  Langchain_text_splitters: Quebrar os PDFs em pequenos pedaços 
#faiss-cpu: Busca similaridade e agrupamento de vetores.

#recurso do Langchain splitters (RecursiveCharacterTexSplit..) serve para
#realizar chunkings= fragmenta um grande conj. de dados em partes menores
#aqui tbm vou definir um overlap=sobreposição, p/ que o final de um chunking seja o inicio do proximo
#pega um pouco do contexto e pouco do próximo chunking..
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
#import do Generative AI p/ embeddings

from langchain_community.vectorstores import FAISS
#agora é usar os embeddings pra fazer calculo de similaridade, usando a biblioteca faiss

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
#essa funcao pega todos os documentos e utiliza como contexto do prompt

from typing import TypedDict, Optional
#importação da biblioteca que me ajudará

from langgraph.graph import StateGraph, START, END

#pra gerar o desenho grafico
from IPython.display import display, Image

secrets = toml.load("secrets.toml")
api_key_gemini = secrets["default"]["GEMINI_API_KEY"]
docs = []

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0,
    api_key=api_key_gemini
    #verbose=True
)

pergunta = input("faça uma pergunta a IA: ")
resposta_teste = llm.invoke(pergunta)
print(resposta_teste.content)


TRIAGEM_PROMPT_ALURA = (
    "Você é um triador de Service Desk para políticas internas da empresa Carraro Desenvolvimento. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
    "{\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
    '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n"
    "Regras:\n"
    '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas (Ex: "Posso reembolsar a internet do meu home office?", "Como funciona a política de alimentação em viagens?").\n'
    '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema ou contexto (Ex: "Preciso de ajuda com uma política", "Tenho uma dúvida geral").\n'
    '- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou acesso especial, ou quando o usuário explicitamente pede para abrir um chamado (Ex: "Quero exceção para trabalhar 5 dias remoto.", "Solicito liberação para anexos externos.", "Por favor, abra um chamado para o RH.").'
    "Analise a mensagem e decida a ação mais apropriada."
)

class TriagemOut(BaseModel):
  decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
  urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
  campos_faltantes: List[str] = Field(default_factory=list)

llm_triagem = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=1.0,
    api_key=api_key_gemini,
)
#configuranbdo o llm_triagem pra dados estruturados junto ao TriagemOut
triagem_chain = llm_triagem.with_structured_output(TriagemOut)

def triagem(mensagem_humana: str) -> Dict:
    saida: TriagemOut = triagem_chain.invoke([
       SystemMessage(content=TRIAGEM_PROMPT_ALURA),
       HumanMessage(content=mensagem_humana)
    ]) 

    return saida.model_dump()

testes = ["Posso reembolsar a internet?",
          "Quero mais 5 dias de trabalho remoto, como faço?",
          "Quantas capivaras tem no Rio Pinheiros?"]
for msg_teste in testes:
   print(f"Pergunta: {msg_teste} \n > Resposta: {triagem(msg_teste)}\n")

#O Path é o caminho da onde salvei os documentos para o RAG
# o .glob é pra definir que só quero ler os arquivos .pdf dentro da pasta
#o try é pra tentar carregar os arquivos
#depois chamo a lista docs para extrair
#Obs.: Caso eu troque a pasta de lugar, usar: base_dir = Path(__file__).parent & pdf_dir = base_dir / "pdf-agent-development"
for n in Path("pdf-agent-development").glob("*.pdf"):
    try:
        loader = PyMuPDFLoader(str(n))
        docs.extend(loader.load())
        print(f"Arquivo carregado com sucesso {n.name}.")
    except Exception as e:
        print(f"Erro ao carregar o arquivo {n.name}: {e}.")
print(f"Total de documentos carregados: {len(docs)}.")

#var pra definir o chunking + overlap
#tamanho do chunk=300 caracteres.. o Overlap=30 (pra manter contexto)
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

#aqui ele respeita os parametros de 'splitter' e realiza o split na lista 'docs'
chunks = splitter.split_documents(docs)


#quero realizar uma busca, e de acordo com a busca, quero entender
#dentro destes chunks a próximidade das coisas 'assuntos'=transformar 
# em vetores

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key_gemini 
)

#agora é usar os embeddings pra fazer 
# calculo de similaridade, usando a biblioteca faiss

#agora é usar os embeddings pra fazer calculo de similaridade, usando a biblioteca faiss

vectorstores = FAISS.from_documents(chunks, embeddings)

#agora criar o R rag-> Retrieval
#cada pergunta do usuario, também é tranformada num vetor de embeddings, e a partir disto, posso
#escolher entre os 3 mais similares p/ retornar (ou 1 ou mais similares)

#aqui vamos configurar o tipo de busca, nota de corte e similaridade
retriever = vectorstores.as_retriever(search_type="similarity_score_threshold", 
                                      search_kwargs={"score_threshold":0.3, "k":4})#0.3 é padrão, mas aqui eu posso aumentar o limite para retornar mais chunks

prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
    "Você é um assistente de políticas internas (RH/IT) da empresa FiaresDev."
    "Responda SOMENTE com base no contexto fornecido."
    "Se não houver base suficiente, responda apenas 'Não sei'."),
    #pergunta do humano.
    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])
#posso criar outro llm pra test tbm
document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)

def perguntar_politica_RAG(pergunta: str) -> Dict:
    docs_relacionados = retriever.invoke(pergunta)
    #se não tem algum doc relacionado pro RAG, ele ja responde que não sabe
    if not docs_relacionados:
        return{"answer": "Não sei.",
            "citacoes": [],
            "contexto_encontrado": False}
 
    answer = document_chain.invoke({"input": pergunta, 
                                  "context": docs_relacionados})
    #aqui eu chamo o llm e limpo a resposta dele
    txt = (answer or "").strip()

    if txt.rstrip(".!?") == "Não sei":
        return{"answer": "Não sei.",
            "citacoes": [],
            "contexto_encontrado": False}
    
    return{"answer": txt,
        "citacoes": docs_relacionados,
        "contexto_encontrado": True}

testes = ["Posso reembolsar a internet?",
          "Quero mais 5 dias de trabalho remoto, como faço?",
          "Quantas capivaras tem no Rio Pinheiros?"]

for msg_teste in testes:
  resposta = perguntar_politica_RAG(msg_teste)
  print(f"Pergunta: {msg_teste}")
  print(f"Resposta: {resposta['answer']}")
  if resposta['contexto_encontrado']:
      print("CITAÇÕES:")
      print(resposta['citacoes'])
      print("--------------------------\n")

class AgentState(TypedDict, total = False):
  mensagem: str
  triagem: dict
  resposta: Optional[str]
  citacoes: List[dict]
  rag_sucesso: bool
  acao_final: str

def node_triagem(state: AgentState) -> AgentState:
    print("Executando nó de triagem...")
    return {"triagem": triagem(state["pergunta"])}

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

#poderia conectar isto a um email / ferramenta(Jira, por ex.)
    return {
        "resposta": f"Abrindo chamado com urgência {triagem['urgencia']}. Descrição: {state['pergunta'][:140]}",
        "citacoes": [],
        "acao_final": "ABRIR_CHAMADO"
    }

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

    return "info-pedir"

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


graph_bytes = grafo.get_graph().draw_mermaid_png()
display(Image(graph_bytes))