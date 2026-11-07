# Atividade 05 TESI I - Sistema de Recuperação com VectorDB e RetrievalQA

Repositório desenvolvido para a Avaliação Online 5 da disciplina Tópicos Específicos em Sistemas para Internet I (IFRN - TSI).

## Objetivo
Implementar um pipeline funcional de **RAG (Retrieval-Augmented Generation)** utilizando LangChain, FAISS como banco vetorial local e o módulo `RetrievalQA` para responder a perguntas contextuais sobre Redes Neurais e Inteligência Artificial.

## Arquitetura do Pipeline
1. **Corpus Textual:** Conceitos de Redes Neurais Artificiais, arquitetura Transformer (mecanismo de autoatenção) e RAG.
2. **Chunking:** `RecursiveCharacterTextSplitter` (chunk_size: 250, chunk_overlap: 50), gerando 9 chunks em memória.
3. **Embeddings:** Projeção semântica local via `nomic-embed-text` no Ollama.
4. **Base Vetorial:** Indexação vetorial de alta performance com `FAISS`.
5. **RetrievalQA:** Recuperação dos top-2 chunks por similaridade de cosseno e geração de resposta contextualizada com o modelo `llama3.2:latest`.

## Pré-requisitos
- [Ollama](https://ollama.com/) instalado e em execução no Windows.
- Python 3.10 ou superior.
- Modelos baixados no Ollama:
  ```powershell
  ollama pull nomic-embed-text
  ollama pull llama3.2
  ```

## Configuração do Ambiente

1. **Clonar o repositório:**
   ```powershell
   git clone [https://github.com/fheinrich/tsi-rag-faiss.git](https://github.com/fheinrich/tsi-rag-faiss.git)
   cd tsi-rag-faiss
   ```

2. **Criar e ativar o ambiente virtual (venv):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Instalar dependências:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Executar o script:**
   ```powershell
   python rag_app.py
   ```

## Exemplos de Consultas e Respostas Reais do Console

### Pergunta 1: *'Qual é o elemento principal da arquitetura Transformer?'*
- **Resposta LLM:** O elemento principal da arquitetura Transformer é o mecanismo de autoatenção (self-attention).
- **Chunk Recuperado:** `"O elemento principal do Transformer é o mecanismo de autoatenção (self-attention), que permite ao modelo ponderar dinamicamente a importância de cada palavra em relação a todas as outras na mesma sequência..."`

### Pergunta 2: *'O que é RAG e como ele ajuda a mitigar alucinações?'*
- **Resposta LLM:** RAG (Retrieval-Augmented Generation) é uma técnica que combina a recuperação de dados externos com a geração de texto. Ela ajuda a mitigar alucinações fornecendo ao modelo de linguagem um conjunto de informações mais preciso e relevante, evitando respostas baseadas em informações imprecisas.
- **Chunk Recuperado:** `"Quando uma pergunta é feita, o Retriever localiza os chunks mais relevantes por similaridade geométrica de cosseno e injeta esse conteúdo como contexto no LLM, mitigando alucinações e garantindo respostas fundamentadas."`

### Pergunta 3: *'Como ocorre o aprendizado em uma rede neural profunda?'*
- **Resposta LLM:** O aprendizado em uma rede neural profunda ocorre pelo ajuste iterativo dos pesos sinápticos através de algoritmos de retropropagação (backpropagation) e gradiente descendente.
- **Chunk Recuperado:** `"O aprendizado em uma rede neural profunda ocorre pelo ajuste iterativo dos pesos sinápticos através de algoritmos de retropropagação (backpropagation) e gradiente descendente."`