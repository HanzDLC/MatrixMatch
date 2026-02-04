# pip install sentence-transformers nltk

from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download("punkt_tab")
import re

# Download NLTK punkt for sentence splitting (run once)
nltk.download('punkt')

# 1. Load SBERT
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Keywords (as full phrases)
keywords = [
    "AI chatbot",
    "vector database",
    "natural language processing",
    "real-time communication",
    "software quality evaluation",
]

# 3. Your abstract (with mojibake fixed)
raw_abstract = """
This study presents Tradeansbot in a time where rapid and reliable information accessibility is crucial, 
an artificial intelligence (AI) powered chatbot smartphone application created for Iloilo Science and 
Technology (ISAT-U) that offers precise, real-time answers to questions about the university. With the help 
of ChromaDB, a vector database that facilitates effective information retrieval via vector embeddings and 
cosine similarity, TRADEANSBOT incorporates OpenAI's GPT-based language model for natural language 
interpretation and answer creation. LangChain, which organizes the communication between the language model 
and the document retriever, is used to process user inputs. FastAPI, a high-performance web framework, is 
used in the development of the backend to manage real-time communication between the server and mobile 
application. Supabase is employed for student account authentication and data management, ensuring secure 
and personalized access. With Supabase handling real-time data and user authentication, the application was 
developed with Flutter for the mobile interface and FastAPI for backend processing. The integration of 
external data and query flow were also coordinated using LangChain. The system was evaluated using the 
ISO 25010 software quality model, focusing on criteria such as functionality, usability, reliability, 
performance efficiency, maintainability, security, and portability. Evaluation results from IT professionals 
and users demonstrated that TRADEANSBOT performed excellently in most areas, highlighting its potential to 
serve as a dependable academic support tool.
"""

def clean_text(text: str) -> str:
    return (text
            .replace("â€™", "’")
            .replace("â€œ", "“")
            .replace("â€", "”")
            .replace("Â", "")
            .strip())

abstract = clean_text(raw_abstract)

# 4. Split abstract into sentences
sentences = nltk.sent_tokenize(abstract)

# 5. Encode all sentences once
sent_embeddings = model.encode(sentences, convert_to_tensor=True)

def semantic_keyword_scores(keywords, sentences, sent_embeddings):
    scores = []

    for kw in keywords:
        kw_emb = model.encode(kw, convert_to_tensor=True)
        sims = util.cos_sim(kw_emb, sent_embeddings)[0]
        max_sim = float(sims.max().item())
        scores.append((kw, max_sim))

    return scores

scores = semantic_keyword_scores(keywords, sentences, sent_embeddings)

# Convert cosine similarity to percentage
def scale_cosine_to_percent(cos_sim: float) -> float:
    x = (cos_sim - 0.2) / 0.6
    x = max(0.0, min(1.0, x))
    return x * 100.0

print("SBERT phrase relevance (max similarity):\n")
for kw, max_sim in scores:
    max_pct = scale_cosine_to_percent(max_sim)
    print(f"{kw:28s} -> max_sim={max_sim:.4f}, percent={max_pct:6.2f}%")
