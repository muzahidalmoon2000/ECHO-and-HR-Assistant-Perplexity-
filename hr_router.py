
import os
import json
import requests
import docx
from PyPDF2 import PdfReader

# Load Perplexity API Key
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
PPLX_API_URL = "https://api.perplexity.ai/chat/completions"

HR_KB_DIR = os.path.join("knowledge_base", "documents")
HR_KB_JSON = os.path.join("knowledge_base", "hr_knowledge.json")


def call_perplexity_chat(system_prompt, user_input, temperature=0.2):
    headers = {
        "Authorization": f"Bearer {PPLX_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        "temperature": temperature
    }

    response = requests.post(PPLX_API_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content'].strip()


def classify_intent(user_query):
    system_prompt = (
        "You are an intent classification assistant for an HR assistant system. "
        "Classify the user input into one of these intents:\n"
        "- 'nid_info': NID-related questions (e.g., NID number, national ID)\n"
        "- 'leave_balance': Leave status or how much leave they have\n"
        "- 'leave_policy': Leave rules or company policy on leave/holidays\n"
        "- 'hr_admin': Any other HR document-based request\n"
        "- 'general': Anything else not HR related\n\n"
        "Reply in this JSON format: {\"intent\": \"intent_name\"}"
    )
    response = call_perplexity_chat(system_prompt, user_query, temperature=0)
    try:
        return json.loads(response)["intent"]
    except:
        return "general"


def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as e:
        print(f"❌ PDF extract failed: {file_path} - {e}")
        return ""


def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except Exception as e:
        print(f"❌ DOCX extract failed: {file_path} - {e}")
        return ""


def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"❌ TXT extract failed: {file_path} - {e}")
        return ""


def build_hr_knowledge_json():
    """Extract all HR documents and save them into a single JSON file."""
    os.makedirs(HR_KB_DIR, exist_ok=True)
    knowledge = {}

    for fname in os.listdir(HR_KB_DIR):
        fpath = os.path.join(HR_KB_DIR, fname)
        if fname.lower().endswith(".pdf"):
            text = extract_text_from_pdf(fpath)
        elif fname.lower().endswith(".docx"):
            text = extract_text_from_docx(fpath)
        elif fname.lower().endswith(".txt"):
            text = extract_text_from_txt(fpath)
        else:
            continue

        if text:
            knowledge[fname] = text

    with open(HR_KB_JSON, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, indent=2, ensure_ascii=False)


def load_knowledge_context():
    if not os.path.exists(HR_KB_JSON):
        return ""
    try:
        with open(HR_KB_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return "\n\n".join(data.values())
    except Exception as e:
        print(f"⚠️ Failed to load HR knowledge: {e}")
        return ""


def search_hr_knowledge_base(user_query):
    json_path = os.path.join("knowledge_base", "hr_knowledge.json")
    if not os.path.exists(json_path):
        return "⚠️ HR knowledge base is missing."

    with open(json_path, "r", encoding="utf-8") as jf:
        data = json.load(jf)

    # Combine all content into one large context block
    combined_context = "\n\n".join(data.values())
    return generate_answer_from_context(user_query, combined_context)


def generate_answer_from_context(user_query, context):
    system_prompt = (
        "You are a helpful HR assistant. Use the provided document contents to answer the user's question. "
        "Only answer using the given context. If the answer isn’t in the documents, say you don’t know."
    )
    full_prompt = f"User question: {user_query}\n\nDocument contents:\n{context}"
    return call_perplexity_chat(system_prompt, full_prompt, temperature=0.3)


def handle_query(user_query,intent=None):
    intent = classify_intent(user_query)

    if intent in {"nid_info", "leave_balance", "leave_policy", "hr_admin"}:
        return search_hr_knowledge_base(user_query)

    return "🤖 I'm not sure how to answer that. Please ask something HR-related."
