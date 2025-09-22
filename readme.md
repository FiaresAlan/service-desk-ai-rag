# 🤖 Agente de Triagem com RAG + LangChain + Google Gemini

Este projeto implementa um agente inteligente para **triagem de chamados internos** em um Service Desk.  
Ele utiliza **LLMs do Google Gemini** integrados ao **LangChain**, com suporte a **RAG (Retrieval Augmented Generation)** e **fluxo de decisão via LangGraph**.

---

## 🚀 Funcionalidades
- **Triagem Automática** com três possibilidades:
  - `AUTO_RESOLVER`: responde automaticamente com base nas políticas internas.
  - `PEDIR_INFO`: solicita informações adicionais do usuário.
  - `ABRIR_CHAMADO`: abre um chamado (ex.: exceções, aprovações, acessos especiais).
- **RAG (Retrieval Augmented Generation)**:
  - Carrega e processa documentos PDF com políticas internas.
  - Fragmenta em chunks com sobreposição de contexto.
  - Indexa vetores usando **FAISS** para busca semântica.
- **Fluxo de Estados (LangGraph)**:
  - Garante decisões encadeadas e rastreáveis.
  - Exibe grafo do fluxo de decisão.

---

## 📂 Estrutura do Projeto
📦 projeto-triagem
┣ 📂 pdf-agent-development # PDFs com políticas internas
┣ 📜 main.py # código principal
┣ 📜 requirements.txt # dependências
┣ 📜 secrets.toml # chaves da API (não versionar!)
┗ 📜 README.md # este arquivo


---

## Pré-requisitos
- **Python 3.10+**
- Conta no **Google AI Studio** para obter a chave do Gemini.

Crie o arquivo `secrets.toml` na raiz do projeto:

```toml
[default]
GEMINI_API_KEY = "sua_chave_api_aqui"

---

## Como Executar

1. Criar ambiente virtual:

python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

2. Instalar dependências:

pip install -r requirements.txt

3. Executar o projeto:

python app.py

---

## Exemplo de Uso
faça uma pergunta a IA: Posso reembolsar a internet?
> Resposta: {"decisao": "AUTO_RESOLVER", "urgencia": "BAIXA", "campos_faltantes": []}

faça uma pergunta a IA: Quero mais 5 dias de trabalho remoto, como faço?
> Resposta: {"decisao": "ABRIR_CHAMADO", "urgencia": "MEDIA", "campos_faltantes": []}

faça uma pergunta a IA: Quantas capivaras tem no Rio Pinheiros?
> Resposta: Não sei.

-----

Tecnologias Utilizadas

LangChain
 → orquestração da IA

Google Generative AI (Gemini)
 → modelo de LLM e embeddings

LangGraph
 → fluxo de estados

FAISS
 → armazenamento vetorial

PyMuPDF
 → leitura de PDFs
------

Avisos Importantes para quem utilizar o reposotório:

O arquivo secrets.toml não deve ser commitado.

Os documentos para o RAG devem estar na pasta pdf-agent-development/.

Certifique-se de ter dependências instaladas corretamente no ambiente virtual.