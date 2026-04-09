import streamlit as st
from main import setup_rag, ask_question

st.set_page_config(
    page_title="RAG Document QA Dashboard",
    page_icon="📄",
    layout="wide"
)

@st.cache_resource
def init_rag():
    return setup_rag("data")

st.title("📄 RAG Document QA Dashboard")
st.caption("Ask questions about your PDF documents and get grounded answers with sources.")

try:
    rag_info = init_rag()

    st.sidebar.title("System Info")
    st.sidebar.write(f"**Loaded documents:** {rag_info['doc_count']}")
    st.sidebar.write(f"**Total chunks:** {rag_info['chunk_count']}")
    st.sidebar.write("**Model:** gpt-4o-mini")
    st.sidebar.write("**Embedding:** text-embedding-3-small")
    st.sidebar.write("**Retrieval:** MMR")
    st.sidebar.write("**Top k:** 6")

    question = st.text_input("Ask a question:")

    if st.button("Ask Question"):
        if question.strip():
            with st.spinner("Searching documents and generating answer..."):
                result = ask_question(question)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("## 💬 Answer")
                st.write(result["answer"])

                st.markdown("## 📄 Sources")
                if result["sources"]:
                    for src in result["sources"]:
                        st.markdown(f"- {src}")
                else:
                    st.write("No sources available.")

            with col2:
                st.markdown("## 📊 Confidence")
                if result["confidence"] == "High":
                    st.success(result["confidence"])
                elif result["confidence"] == "Medium":
                    st.warning(result["confidence"])
                else:
                    st.error(result["confidence"])

                st.markdown("## 🔍 Grounded")
                if result["grounded"] == "YES":
                    st.success("YES")
                else:
                    st.error("NO")
        else:
            st.warning("Please enter a question.")

except Exception as e:
    st.error(f"UI initialization failed: {e}")