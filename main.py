import os
import base64
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END


# =========================
# 1. Load environment
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

openai_client = OpenAI(api_key=OPENAI_API_KEY)


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
        raise ValueError("RAG application is not initialized. Please run setup_rag() first.")
    return rag_app


# =========================
# 3. Define graph state
# =========================
class GraphState(TypedDict):
    question: str
    documents: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    confidence: str
    grounded: str


# =========================
# 4. Initialize models
# =========================
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# =========================
# 5. File types and settings
# =========================
PDF_EXTENSIONS = {".pdf"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}

FRAME_INTERVAL_SECONDS = 15
MAX_FRAMES_PER_VIDEO = 12
TRANSCRIPT_CHUNK_SIZE = 1200
TRANSCRIPT_CHUNK_OVERLAP = 150
GENERAL_CHUNK_SIZE = 800
GENERAL_CHUNK_OVERLAP = 120


# =========================
# 6. Utility functions
# =========================
def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "ffmpeg is not installed or not available in PATH. "
            "Please install ffmpeg before processing video files."
        )


def format_seconds(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def split_text_to_documents(
    text: str,
    source: str,
    modality: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = TRANSCRIPT_CHUNK_SIZE,
    chunk_overlap: int = TRANSCRIPT_CHUNK_OVERLAP,
) -> List[Document]:
    text = (text or "").strip()
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)

    docs: List[Document] = []
    for idx, chunk in enumerate(chunks, start=1):
        metadata = {
            "source": source,
            "modality": modality,
            "chunk_index": idx
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        docs.append(Document(
            page_content=chunk,
            metadata=metadata
        ))

    return docs


# =========================
# 7. PDF loading
# =========================
def load_pdf_documents(pdf_dir: str) -> List[Document]:
    if not os.path.exists(pdf_dir):
        return []

    loader = PyPDFDirectoryLoader(pdf_dir)
    docs = loader.load()

    normalized_docs: List[Document] = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "unknown")

        normalized_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": source,
                    "modality": "pdf",
                    "page": page
                }
            )
        )

    return normalized_docs


# =========================
# 8. Audio loading
# =========================
def transcribe_audio_file(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        transcript = openai_client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f
        )

    text = getattr(transcript, "text", None)

    if isinstance(text, str) and text.strip():
        return text.strip()

    return str(transcript).strip()


def load_audio_documents(audio_dir: str) -> List[Document]:
    if not os.path.exists(audio_dir):
        return []

    docs: List[Document] = []

    for filename in sorted(os.listdir(audio_dir)):
        ext = Path(filename).suffix.lower()
        if ext not in AUDIO_EXTENSIONS:
            continue

        file_path = os.path.join(audio_dir, filename)
        print(f"[Audio] Transcribing file: {filename}")

        transcript_text = transcribe_audio_file(file_path)

        audio_docs = split_text_to_documents(
            text=transcript_text,
            source=filename,
            modality="audio",
            extra_metadata={"original_file": filename}
        )
        docs.extend(audio_docs)

    return docs


# =========================
# 9. Video processing helpers
# =========================
def extract_audio_from_video(video_path: str, output_audio_path: str):
    check_ffmpeg()

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "mp3",
        output_audio_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def extract_frames_from_video(
    video_path: str,
    frame_dir: str,
    interval_seconds: int = FRAME_INTERVAL_SECONDS
):
    check_ffmpeg()
    os.makedirs(frame_dir, exist_ok=True)

    output_pattern = os.path.join(frame_dir, "frame_%04d.jpg")
    fps_expr = f"fps=1/{interval_seconds}"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vf", fps_expr,
        output_pattern
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def summarize_image(image_path: str) -> str:
    b64 = encode_image_to_base64(image_path)

    response = openai_client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Describe this video frame for retrieval and question answering. "
                        "Include visible text, people, actions, objects, charts, scene context, "
                        "and any important on-screen information. Be factual and concise."
                    )
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64}"
                }
            ]
        }]
    )

    return response.output_text.strip()


def load_video_documents(video_dir: str) -> List[Document]:
    if not os.path.exists(video_dir):
        return []

    docs: List[Document] = []

    for filename in sorted(os.listdir(video_dir)):
        ext = Path(filename).suffix.lower()
        if ext not in VIDEO_EXTENSIONS:
            continue

        video_path = os.path.join(video_dir, filename)
        print(f"[Video] Processing file: {filename}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_audio = os.path.join(tmpdir, "audio.mp3")
            frame_dir = os.path.join(tmpdir, "frames")

            # Step 1: Extract audio and transcribe it
            try:
                extract_audio_from_video(video_path, tmp_audio)
                transcript_text = transcribe_audio_file(tmp_audio)

                video_audio_docs = split_text_to_documents(
                    text=transcript_text,
                    source=filename,
                    modality="video_audio",
                    extra_metadata={"original_file": filename}
                )
                docs.extend(video_audio_docs)

            except Exception as e:
                print(f"[Video] Audio extraction or transcription failed for {filename}: {e}")

            # Step 2: Extract frames and summarize them
            try:
                extract_frames_from_video(video_path, frame_dir, FRAME_INTERVAL_SECONDS)

                frame_files = sorted(
                    f for f in os.listdir(frame_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                )[:MAX_FRAMES_PER_VIDEO]

                for idx, frame_file in enumerate(frame_files, start=1):
                    frame_path = os.path.join(frame_dir, frame_file)
                    frame_time_seconds = (idx - 1) * FRAME_INTERVAL_SECONDS
                    time_range = format_seconds(frame_time_seconds)

                    summary = summarize_image(frame_path)

                    docs.append(Document(
                        page_content=summary,
                        metadata={
                            "source": filename,
                            "modality": "video_frame",
                            "frame_file": frame_file,
                            "time_range": time_range,
                            "chunk_index": idx
                        }
                    ))

            except Exception as e:
                print(f"[Video] Frame extraction or vision summarization failed for {filename}: {e}")

    return docs


# =========================
# 10. Load all supported files
# =========================
def load_all_documents(base_data_dir: str = "data") -> List[Document]:
    pdf_dir = os.path.join(base_data_dir, "pdf")
    audio_dir = os.path.join(base_data_dir, "audio")
    video_dir = os.path.join(base_data_dir, "video")

    all_docs: List[Document] = []

    pdf_docs = load_pdf_documents(pdf_dir)
    audio_docs = load_audio_documents(audio_dir)
    video_docs = load_video_documents(video_dir)

    all_docs.extend(pdf_docs)
    all_docs.extend(audio_docs)
    all_docs.extend(video_docs)

    if not all_docs:
        raise ValueError(
            f"No supported files found under {base_data_dir}/pdf, "
            f"{base_data_dir}/audio, or {base_data_dir}/video."
        )

    return all_docs


# =========================
# 11. Split documents for embeddings
# =========================
def split_documents(documents: List[Document]) -> List[Document]:
    if not documents:
        raise ValueError("No documents were provided for splitting.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=GENERAL_CHUNK_SIZE,
        chunk_overlap=GENERAL_CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


# =========================
# 12. Create vector store
# =========================
def create_vectorstore(chunks: List[Document]):
    if not chunks:
        raise ValueError("No chunks were created from the provided files.")
    return FAISS.from_documents(chunks, embeddings)


# =========================
# 13. Retrieval node
# =========================
def retrieve_node(state: GraphState):
    current_retriever = get_retriever()
    question = state["question"]

    results = current_retriever.invoke(question)

    retrieved_docs: List[Dict[str, Any]] = []

    for doc in results:
        metadata = doc.metadata or {}

        retrieved_docs.append({
            "source": os.path.basename(metadata.get("source", "unknown")),
            "modality": metadata.get("modality", "unknown"),
            "page": metadata.get("page"),
            "frame_file": metadata.get("frame_file"),
            "time_range": metadata.get("time_range"),
            "chunk_index": metadata.get("chunk_index"),
            "content": doc.page_content.strip()
        })

    return {
        "question": question,
        "documents": retrieved_docs
    }


# =========================
# 14. Generation node
# =========================
def generate_node(state: GraphState):
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {
            "question": question,
            "documents": [],
            "answer": "I could not find the answer in the provided files.",
            "sources": [],
            "confidence": "Low"
        }

    context_parts = []
    for i, doc in enumerate(documents, 1):
        meta_parts = [
            f"Source: {doc.get('source', 'unknown')}",
            f"Type: {doc.get('modality', 'unknown')}"
        ]

        if doc.get("page") is not None:
            meta_parts.append(f"Page: {doc['page']}")

        if doc.get("frame_file"):
            meta_parts.append(f"Frame: {doc['frame_file']}")

        if doc.get("time_range"):
            meta_parts.append(f"Time: {doc['time_range']}")

        if doc.get("chunk_index") is not None:
            meta_parts.append(f"Chunk: {doc['chunk_index']}")

        metadata_str = ", ".join(meta_parts)

        context_parts.append(
            f"""Document {i}
{metadata_str}
Content: {doc['content']}"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a professional AI assistant specialized in document-based question answering.

You must ONLY use the provided context to answer the question.

Tasks:
1. Answer the question using only the provided context.
2. List the most relevant sources used for the answer.
3. Assign a confidence level:
   - High: directly and clearly supported by the context
   - Medium: mostly supported but somewhat incomplete or indirect
   - Low: weakly supported, unclear, or uncertain

Rules:
- Do NOT use outside knowledge.
- Do NOT guess or invent details.
- If the answer is not supported by the context, say:
  "I could not find the answer in the provided files."
- Be concise, accurate, and specific.
- Clearly reflect whether the source came from a PDF page, an audio chunk, a video audio chunk, or a video frame.

Return your response in exactly this format:

Answer: <your answer>

Sources:
- <source details>
- <source details>

Confidence: <High | Medium | Low>

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    output = response.content.strip()

    answer = "I could not find the answer in the provided files."
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
# 15. Grounding / fact-check node
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
        meta_parts = [
            f"Source: {doc.get('source', 'unknown')}",
            f"Type: {doc.get('modality', 'unknown')}"
        ]

        if doc.get("page") is not None:
            meta_parts.append(f"Page: {doc['page']}")

        if doc.get("frame_file"):
            meta_parts.append(f"Frame: {doc['frame_file']}")

        if doc.get("time_range"):
            meta_parts.append(f"Time: {doc['time_range']}")

        context_parts.append(
            f"{', '.join(meta_parts)}\nContent: {doc['content']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a strict factual verifier.

Determine whether the answer is fully supported by the given context.

Rules:
- Reply ONLY with: YES or NO
- YES = every part of the answer is supported by the context
- NO = any part is unsupported, inferred without evidence, or fabricated

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
# 16. Final node
# =========================
def final_node(state: GraphState):
    if state["grounded"] != "YES":
        return {
            "question": state["question"],
            "documents": state["documents"],
            "answer": "I could not find a fully grounded answer in the provided files.",
            "sources": [],
            "confidence": "Low",
            "grounded": state["grounded"]
        }

    return state


# =========================
# 17. Build LangGraph workflow
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
# 18. Setup RAG system
# =========================
def setup_rag(base_data_dir: str = "data"):
    raw_docs = load_all_documents(base_data_dir)
    chunks = split_documents(raw_docs)
    vectorstore = create_vectorstore(chunks)

    local_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20}
    )

    set_retriever(local_retriever)

    app = build_graph()
    set_app(app)

    modality_counts: Dict[str, int] = {}
    for doc in raw_docs:
        modality = doc.metadata.get("modality", "unknown")
        modality_counts[modality] = modality_counts.get(modality, 0) + 1

    return {
        "app": app,
        "doc_count": len(raw_docs),
        "chunk_count": len(chunks),
        "modality_counts": modality_counts
    }


# =========================
# 19. Ask a question
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
# 20. Main CLI
# =========================
if __name__ == "__main__":
    try:
        info = setup_rag("data")
        print(f"Loaded document units: {info['doc_count']}")
        print(f"Created chunks: {info['chunk_count']}")
        print("Loaded by modality:")
        for modality, count in info["modality_counts"].items():
            print(f"  - {modality}: {count}")

        print("\nMultimodal RAG system is ready.")
        print("Supported folders:")
        print("  data/pdf")
        print("  data/audio")
        print("  data/video")
        print("Type 'exit' to quit.\n")

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
                print("- No sources found")

            print(f"\nConfidence Level: {result['confidence']}")
            print(f"Grounded (Fact-Checked): {result['grounded']}")
            print("===================================\n")

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting...")

    except Exception as e:
        print("Error:", e)