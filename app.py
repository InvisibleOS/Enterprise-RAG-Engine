import streamlit as st
import os
import tempfile
from rag_engine import EnterpriseBrain

# 1. Page Configuration
st.set_page_config(
    page_title="Enterprise Knowledge Brain",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Enterprise RAG Brain")
st.markdown("### Powered by Google Gemini & Pinecone")

# 2. Sidebar for Setup
with st.sidebar:
    st.header("Configuration")
    
    # Try to load keys from secrets (if they exist), otherwise ask user
    if "GOOGLE_API_KEY" in st.secrets:
        google_key = st.secrets["GOOGLE_API_KEY"]
        st.success("Google Key loaded from Secrets!")
    else:
        google_key = st.text_input("Google API Key", type="password")

    if "PINECONE_API_KEY" in st.secrets:
        pinecone_key = st.secrets["PINECONE_API_KEY"]
        st.success("Pinecone Key loaded from Secrets!")
    else:
        pinecone_key = st.text_input("Pinecone API Key", type="password")
    
    st.divider()
    
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.brain = None
        st.rerun()

# 3. Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "brain" not in st.session_state:
    st.session_state.brain = None

# 4. Handle File Upload
if uploaded_file and google_key and pinecone_key:
    # If the brain isn't started, start it
    if st.session_state.brain is None:
        st.session_state.brain = EnterpriseBrain(google_key)
        
        with st.spinner("Processing PDF (Ingesting & Vectorizing)..."):
            # Save uploaded file to a temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
                
            # Run the ingestion
            status_msg = st.session_state.brain.ingest_pdf(tmp_path, pinecone_key)
            st.success(status_msg)
            os.remove(tmp_path) # Clean up file

# 5. Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. User Input & Response Generation
if prompt := st.chat_input("Ask a question about your document..."):
    # Checks
    if not google_key or not pinecone_key:
        st.error("Please enter your API Keys in the sidebar.")
        st.stop()
    if not st.session_state.brain:
        st.error("Please upload a document first.")
        st.stop()

    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_data = st.session_state.brain.ask(prompt)
            answer = response_data["result"]
            sources = response_data.get("source_documents", [])
            
            st.markdown(answer)
            
            # Show Citations (The "Resume Boosting" Feature)
            if sources:
                with st.expander("📚 View Source Context"):
                    for idx, doc in enumerate(sources):
                        st.markdown(f"**Source {idx+1} (Page {doc.metadata.get('page', 0) + 1}):**")
                        st.markdown(f"> {doc.page_content}")
                        st.divider()

    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": answer})