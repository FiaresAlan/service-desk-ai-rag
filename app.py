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

secrets = toml.load("secrets.toml")
api_key_gemini = secrets["default"]["GEMINI_API_KEY"]

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
