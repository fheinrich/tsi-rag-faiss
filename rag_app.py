"""
Sistema de Recuperação de Informações com FAISS e RetrievalQA
Disciplina: Tópicos Específicos em Sistemas para Internet I (IFRN - TSI)
Avaliação Online 5
"""

import sys
import warnings

# Silencia avisos de migração no terminal para manter a saída limpa
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Import do RetrievalQA compatível com o LangChain moderno
try:
    from langchain_classic.chains import RetrievalQA
except ImportError:
    from langchain.chains import RetrievalQA

# 1. Base Documental de Estudo (Inteligência Artificial e Redes Neurais)
TEXTO_BASE = """
As Redes Neurais Artificiais (RNAs) são modelos computacionais inspirados na estrutura do cérebro humano.
Elas são compostas por neurônios artificiais organizados em camadas: camada de entrada, camadas ocultas e camada de saída.
O aprendizado em uma rede neural profunda ocorre pelo ajuste iterativo dos pesos sinápticos através de algoritmos de retropropagação (backpropagation) e gradiente descendente.

A arquitetura Transformer, introduzida em 2017 no artigo 'Attention is All You Need', revolucionou o processamento de linguagem natural.
O elemento principal do Transformer é o mecanismo de autoatenção (self-attention), que permite ao modelo ponderar dinamicamente a importância de cada palavra em relação a todas as outras na mesma sequência, viabilizando o processamento paralelo em GPUs.

A técnica de RAG (Retrieval-Augmented Generation) complementa modelos de linguagem combinando recuperação de dados externos com geração de texto.
Em vez de depender apenas da memória paramétrica do modelo, o RAG converte documentos em representações vetoriais densas (embeddings) e os armazena em bases vetoriais como o FAISS.
Quando uma pergunta é feita, o Retriever localiza os chunks mais relevantes por similaridade geométrica de cosseno e injeta esse conteúdo como contexto no LLM, mitigando alucinações e garantindo respostas fundamentadas.
"""

def main():
    print("=" * 70)
    print("INICIALIZANDO O PIPELINE RAG (FAISS + LANGCHAIN + OLLAMA)")
    print("=" * 70)

    # 2. Criação do Documento e Aplicação de Chunking
    documento = Document(page_content=TEXTO_BASE, metadata={"fonte": "apostila_ia"})
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents([documento])
    print(f"\n[1] Texto segmentado em {len(chunks)} chunks com sucesso.")

    # 3. Geração de Embeddings e Armazenamento Vetorial no FAISS
    print("\n[2] Gerando embeddings locais com Ollama ('nomic-embed-text')...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    print("[3] Indexando vetores densos na base vetorial FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("    -> Base FAISS construída com sucesso em memória.")

    # 4. Configuração do Retriever e do Modelo LLM
    print("\n[4] Configurando o Retriever e o LLM ('llama3.2')...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = OllamaLLM(model="llama3.2:latest", temperature=0.1)

    # 5. Montagem da Cadeia de Pergunta e Resposta (RetrievalQA)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    # 6. Testes Práticos de Perguntas e Respostas
    perguntas = [
        "Qual é o elemento principal da arquitetura Transformer?",
        "O que é RAG e como ele ajuda a mitigar alucinações?",
        "Como ocorre o aprendizado em uma rede neural profunda?"
    ]

    print("\n" + "=" * 70)
    print("INICIANDO CONSULTAS VIA RETRIEVALQA")
    print("=" * 70)

    for i, pergunta in enumerate(perguntas, 1):
        print(f"\n--- Pergunta {i}: '{pergunta}' ---")
        resultado = qa_chain.invoke({"query": pergunta})
        
        print("\nResposta gerada pelo LLM com base no FAISS:")
        print(resultado["result"].strip())
        
        print("\nFontes recuperadas da base vetorial:")
        for doc in resultado["source_documents"]:
            trecho = doc.page_content.replace("\n", " ").strip()
            print(f"  * [Chunk]: \"{trecho}\"")
        print("-" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERRO NA EXECUÇÃO]: {e}")
        print("Certifique-se de que o aplicativo Ollama está em execução no Windows.")
        sys.exit(1)