# app/app.py

import streamlit as st
import torch
import faiss
import pickle
import requests
from bs4 import BeautifulSoup

from model import TransformerDecoder
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer

from agents.agent_controller import multi_agent_system

# ----------------- SETUP -----------------

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cache loading components
@st.cache_resource
def load_all_components():
    tokenizer = Tokenizer.from_file("/content/drive/MyDrive/pooosaaaaa/azure_tokenizer/tokenizer.json")
    model = TransformerDecoder(vocab_size=50000)
    model.load_state_dict(torch.load("/content/drive/MyDrive/pooosaaaaa/saved_models/azure_migration_llm.pt", map_location=device))
    model.to(device)
    model.eval()

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("/content/drive/MyDrive/pooosaaaaa/vector_index/faiss_index.bin")
    with open("/content/drive/MyDrive/pooosaaaaa/vector_index/texts.pkl", "rb") as f:
        documents = pickle.load(f)

    return tokenizer, model, embedder, index, documents

tokenizer, model, embedder, index, documents = load_all_components()

# ----------------- FUNCTIONS -----------------

@torch.no_grad()
def search_vector_db(user_query, top_k=3):
    query_embedding = embedder.encode([user_query], convert_to_numpy=True)
    D, I = index.search(query_embedding, top_k)
    retrieved_texts = [documents[i] for i in I[0]]
    return retrieved_texts, D[0]

def scrape_microsoft_learn(user_query):
    """
    Basic scraper that simulates a search by hitting a Microsoft Learn page related to Azure.
    (For real-world, use Bing API or a custom search engine.)
    """
    search_url = "https://learn.microsoft.com/en-us/azure/?product=popular"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract main content from the page
    content_section = soup.find("main")
    paragraphs = []
    if content_section:
        for p in content_section.find_all(["p", "li"]):
            paragraphs.append(p.get_text(strip=True))

    return "\n".join(paragraphs[:20])  # Take first few paragraphs for now

@torch.no_grad()
def generate_final_answer(context, user_query, max_new_tokens=200):
    prompt = f"""
You are an Azure Cloud Migration Expert Assistant.
Given the context and the user's question, first check if critical details are missing.
If missing, ask polite follow-up questions.
Otherwise, give Azure service mapping, estimated costs, and best practice recommendations.

Context:
{context}

User Question:
{user_query}

Answer:
"""
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoding.ids, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        outputs = model(input_ids)
        next_token_logits = outputs[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        input_ids = torch.cat((input_ids, next_token_id), dim=1)

    generated_ids = input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    return generated_text

# ----------------- STREAMLIT UI -----------------

st.set_page_config(page_title="Azure Migration AI Assistant", page_icon="🧠")
st.title("🚀 Azure Migration AI Assistant (Multi-Agent + Live Search)")

st.markdown("""
Ask anything about migrating your app, database, APIs to Azure.  
This assistant uses:
- Local knowledge base
- Live Microsoft Docs scraping
- Multi-agent system for cost estimation, optimization
""")

query = st.text_input("Enter your migration-related question:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Submit"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking... 🤔"):

            # 1. First search locally in Vector DB
            retrieved_texts, similarity_scores = search_vector_db(query, top_k=3)

            # 2. Check if similarity is low -> Do live web scraping
            confidence = similarity_scores[0]  # Best match
            st.write(f"**Vector search confidence:** {round(confidence, 3)}")

            context = "\n\n".join(retrieved_texts)

            if confidence > 0.6:
                # Use local vector db
                st.info("✅ Using local Azure knowledge base...")
            else:
                # Trigger live web scraping
                st.warning("⚡ Vector confidence low, searching Microsoft Learn site...")
                live_context = scrape_microsoft_learn(query)
                context += "\n\n" + live_context

            # 3. Generate final answer
            answer = generate_final_answer(context, query)

            # 4. Save conversation
            st.session_state.chat_history.append((query, answer))

# ----------------- Display Full Conversation -----------------

st.markdown("---")
st.write("### 🗂️ Conversation History:")

for idx, (user, bot) in enumerate(st.session_state.chat_history):
    st.write(f"**You:** {user}")
    st.write(f"**AI:** {bot}")
    st.markdown("---")
