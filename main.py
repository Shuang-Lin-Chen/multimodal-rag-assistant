import os
from typing import TypedDict, List, Dict, Any, Optional

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langgraph.graph import StateGraph, END


# =========================
# 1. Load environment
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")


# =========================
# 2. Global retriever / app
# =========================
retriever = None
rag_app = None


def set_retriever(r):
    global retriever
    retriever = r


def get_retriever():
    global retriever
    if retriever is None:
        raise ValueError("Retriever is not initialized. Please run setup_rag() first.")
    return retriever


def set_app(app):
    global rag_app
    rag_app = app


def get_app():
    global rag_app
    if rag_app is None:
        raise ValueError("RAG app is not initialized. Please run setup_rag() first.")
    return rag_app


# =========================
# 3. Define state
# =========================
class GraphState(TypedDict):
    question: str
    documents: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    confidence: str
    grounded: str


# =========================
# 4. Initialize LLM and embeddings
# =========================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# =========================
# 5. Load and split documents
# =========================
def load_documents(data_path: str = "data"):
    if not os.path.exists(data_path):
        raise ValueError(f"Data folder not found: {data_path}")

    loader = PyPDFDirectoryLoader(data_path)
    docs = loader.load()

    if not docs:
        raise ValueError(f"No PDF documents found in: {data_path}")

    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )
    return splitter.split_documents(documents)


# =========================
# 6. Create vector store
# =========================
def create_vectorstore(chunks):
    if not chunks:
        raise ValueError("No chunks were created from documents.")
    return FAISS.from_documents(chunks, embeddings)


# =========================
# 7. Retriever node
# =========================
def retrieve_node(state: GraphState):
    current_retriever = get_retriever()
    question = state["question"]

    results = current_retriever.invoke(question)

    retrieved_docs: List[Dict[str, Any]] = []

    for doc in results:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "unknown")
        content = doc.page_content.strip()

        retrieved_docs.append({
            "source": source,
            "page": page,
            "content": content
        })

    return {
        "question": question,
        "documents": retrieved_docs
    }


# =========================
# 8. Generator node
# =========================
def generate_node(state: GraphState):
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {
            "question": question,
            "documents": [],
            "answer": "I could not find the answer in the provided documents.",
            "sources": [],
            "confidence": "Low"
        }

    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(
            f"""Document {i}
Source: {doc['source']}
Page: {doc['page']}
Content: {doc['content']}"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document-based question answering assistant.

Use ONLY the provided context.

Your tasks:
1. Answer the user's question using only the context.
2. List the most relevant sources used for the answer.
3. Assign a confidence level:
   - High = answer is directly and clearly supported by the context
   - Medium = answer is mostly supported but somewhat incomplete or indirect
   - Low = answer is weakly supported or uncertain

Rules:
- Do NOT use outside knowledge
- If the answer is not supported, say:
  "I could not find the answer in the provided documents."
- Return your response in exactly this format:

Answer: <answer here>
Sources:
- <source name> (Page <page number>)
- <source name> (Page <page number>)
Confidence: <High/Medium/Low>

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    output = response.content.strip()

    answer = "I could not find the answer in the provided documents."
    sources: List[str] = []
    confidence = "Low"

    lines = output.splitlines()
    current_section: Optional[str] = None
    answer_lines: List[str] = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("Answer:"):
            current_section = "answer"
            answer_text = stripped.replace("Answer:", "", 1).strip()
            if answer_text:
                answer_lines.append(answer_text)

        elif stripped == "Sources:":
            current_section = "sources"

        elif stripped.startswith("Confidence:"):
            current_section = "confidence"
            confidence = stripped.replace("Confidence:", "", 1).strip()

        elif current_section == "answer":
            if stripped:
                answer_lines.append(stripped)

        elif current_section == "sources":
            if stripped.startswith("- "):
                sources.append(stripped[2:].strip())

    if answer_lines:
        answer = " ".join(answer_lines)

    if confidence not in ["High", "Medium", "Low"]:
        confidence = "Low"

    return {
        "question": question,
        "documents": documents,
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }


# =========================
# 9. Grounding check node
# =========================
def hallucination_check_node(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    answer = state["answer"]

    if not documents:
        return {
            "question": question,
            "documents": [],
            "answer": answer,
            "sources": state.get("sources", []),
            "confidence": "Low",
            "grounded": "NO"
        }

    context_parts = []
    for doc in documents:
        context_parts.append(
            f"Source: {doc['source']}, Page: {doc['page']}\nContent: {doc['content']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a strict evaluator.

Determine if the answer is fully supported by the context.

Rules:
- Reply ONLY with: YES or NO
- YES = fully grounded
- NO = contains unsupported or invented information

Context:
{context}

Question:
{question}

Answer:
{answer}

Is the answer fully supported?
"""

    result = llm.invoke(prompt)
    grounded = result.content.strip().upper()

    if grounded not in ["YES", "NO"]:
        grounded = "NO"

    return {
        "question": question,
        "documents": documents,
        "answer": answer,
        "sources": state.get("sources", []),
        "confidence": state.get("confidence", "Low"),
        "grounded": grounded
    }


# =========================
# 10. Final node
# =========================
def final_node(state: GraphState):
    if state["grounded"] != "YES":
        return {
            "question": state["question"],
            "documents": state["documents"],
            "answer": "I could not find a fully grounded answer in the provided documents.",
            "sources": [],
            "confidence": "Low",
            "grounded": state["grounded"]
        }

    return state


# =========================
# 11. Build graph
# =========================
def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("check", hallucination_check_node)
    workflow.add_node("final", final_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "check")
    workflow.add_edge("check", "final")
    workflow.add_edge("final", END)

    return workflow.compile()


# =========================
# 12. Setup RAG
# =========================
def setup_rag(data_path: str = "data"):
    raw_docs = load_documents(data_path)
    chunks = split_documents(raw_docs)
    vectorstore = create_vectorstore(chunks)

    local_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20}
    )

    set_retriever(local_retriever)

    app = build_graph()
    set_app(app)

    return {
        "app": app,
        "doc_count": len(raw_docs),
        "chunk_count": len(chunks)
    }


# =========================
# 13. Ask question
# =========================
def ask_question(question: str):
    app = get_app()

    return app.invoke({
        "question": question,
        "documents": [],
        "answer": "",
        "sources": [],
        "confidence": "",
        "grounded": ""
    })


# =========================
# 14. Main CLI mode
# =========================
if __name__ == "__main__":
    try:
        info = setup_rag("data")
        print(f"Loaded documents: {info['doc_count']}")
        print(f"Created chunks: {info['chunk_count']}")
        print("\nRAG system ready. Type 'exit' to quit.\n")

        while True:
            question = input("Question: ").strip()

            if not question:
                continue

            if question.lower() == "exit":
                print("Goodbye!")
                break

            result = ask_question(question)

            print("\n=========== FINAL RESULT ===========")
            print(f"Answer:\n{result['answer']}\n")

            print("Sources:")
            if result["sources"]:
                for src in result["sources"]:
                    print(f"- {src}")
            else:
                print("- None")

            print(f"\nConfidence: {result['confidence']}")
            print(f"Grounded: {result['grounded']}")
            print("===================================\n")

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting...")

    except Exception as e:
        print("Error:", e)