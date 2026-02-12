'''
RAG system project

GEN AI course

Mohammad AlTa'any

12/27/2025
'''
# Import required libraries

import streamlit as st         # Build the web-based chat interface
import os                      # Access environment variables
from pinecone import Pinecone  # Connect to Pinecone vector database
import requests                # Make HTTP requests to OpenAI APIs
from dotenv import load_dotenv # Load API keys from .env file



# Load environment variables

load_dotenv()  # Load variables from .env file into the environment

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")      # OpenAI API key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Pinecone API key


# Streamlit page configuration

st.set_page_config(
    page_title="RAG Chatbot",  # Browser tab title
    page_icon="🤖",            # Browser tab icon
    layout="centered"          # Page layout
)

# Custom CSS 
st.markdown("""
<style>
    /* Main background - Red gradient */
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d0a0a 100%);
    }
    
    /* Chat messages container */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #DC143C;
    }
    
    /* User message - Darker red */
    .stChatMessage[data-testid="user-message"] {
        background-color: rgba(220, 20, 60, 0.15);
        border-left: 4px solid #DC143C;
    }
    
    /* Assistant message - Lighter background */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: rgba(139, 0, 0, 0.15);
        border-left: 4px solid #8B0000;
    }
    
    /* Input box */
    .stChatInputContainer {
        border-radius: 20px;
        border: 2px solid #DC143C;
    }
    
    /* Title styling with red accent */
    h1 {
        color: #ffffff;
        text-align: center;
        font-family: 'Arial', sans-serif;
        text-shadow: 0 0 20px rgba(220, 20, 60, 0.8);
        padding: 20px 0;
    }
    
    /* Subtitle */
    .stMarkdown p {
        color: #ffcccc;
        text-align: center;
    }
    
    /* Sidebar - Dark red theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0505 0%, #2d0a0a 100%);
        border-right: 2px solid #DC143C;
    }
    
    /* Sidebar header */
    [data-testid="stSidebar"] h2 {
        color: #DC143C;
        font-weight: bold;
    }
    
    /* Info box in sidebar */
    .stAlert {
        background-color: rgba(220, 20, 60, 0.1);
        border: 1px solid #DC143C;
        color: #ffcccc;
    }
    
    /* Buttons - Red theme */
    .stButton button {
        background: linear-gradient(90deg, #DC143C 0%, #8B0000 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(220, 20, 60, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background: linear-gradient(90deg, #FF1744 0%, #DC143C 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(220, 20, 60, 0.5);
    }
    
    /* Spinner color */
    .stSpinner > div {
        border-top-color: #DC143C !important;
    }
    
    /* Text color adjustments */
    .stMarkdown, .stText {
        color: #ffffff;
    }
    
    /* Chat input text */
    .stChatInput input {
        background-color: rgba(220, 20, 60, 0.1);
        color: white;
        border: 1px solid #DC143C;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #DC143C;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #FF1744;
    }
</style>
""", unsafe_allow_html=True)


# Generate text embeddings using OpenAI

def get_embedding(text, api_key):
    """
    Generates an embedding vector for the given text
    using OpenAI's embedding model.
    """
    
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "text-embedding-3-small",
            "input": text
        },
        timeout=30
    )

    # Return None if the request fails
    if response.status_code != 200:
        return None

    return response.json()["data"][0]["embedding"]

def retrieve_context(query, index, api_key, k=10):

    """
    Retrieves the most relevant text chunks from Pinecone
    based on the user's query.
    """

    query_emb = get_embedding(query, api_key)
    if not query_emb:
        return ""
    
    results = index.query(vector=query_emb, top_k=k, include_metadata=True)
    
    if not results or not results.matches:
        return ""
    
    contexts = [match.metadata['text'] for match in results.matches if match.metadata]
    return "\n\n".join(contexts)

def answer_question(question, index, api_key, chat_history):
    """
    Generates an answer using:
    - Retrieved course material
    - Conversation memory
    - OpenAI chat model
    """
    
    context = retrieve_context(question, index, api_key)

    if not context:
        return "❌ No relevant information found in the course materials."

    #Build messages (SYSTEM + MEMORY + CURRENT QUESTION)
    messages = [
        {
            "role": "system",
            "content": """You are a teaching assistant chatbot for a course.

Your role:
- Answer student questions using the provided course materials.
- Use only the information in the course materials.
- If the answer is not fully covered, say:
"I don't know based on the provided course materials."
-answer with the same language in user question
"""
        }
    ]

    # Add previous conversation (memory) so the bot can remember the old questions 
    for msg in chat_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Add current question WITH retrieved context
    messages.append({
        "role": "user",
        "content": f"""Course Materials (Context):
{context}

Current Question:
{question}

Answer:"""
    })

    # Call OpenAI
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o-mini",
            "temperature": 0,
            "messages": messages
        },
        timeout=60
    )

    if response.status_code != 200:
        return f"❌ Error: {response.text}"

    return response.json()["choices"][0]["message"]["content"]

# Initialize Pinecone (cached so it only runs once)
@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index("myind")

# Main app
def main():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image("logo.png", use_container_width=True)
    
    st.title("🤖 Course Assistant Chatbot")
    st.markdown("Ask questions about your course materials!")
    # Initialize Pinecone
    try:
        index = init_pinecone()
    except Exception as e:
        st.error(f"Failed to connect to Pinecone: {e}")
        return
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    
    prompt = st.chat_input("Ask a question about the course...")
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = answer_question(
    prompt,
    index,
    OPENAI_API_KEY,
    st.session_state.messages
)

            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
if __name__ == "__main__":
    main()

