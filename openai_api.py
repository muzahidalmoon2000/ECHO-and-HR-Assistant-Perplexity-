import os
import json
import re
import requests
from dotenv import load_dotenv

load_dotenv()

PPLX_API_KEY = os.getenv("PPLX_API_KEY")
PPLX_API_URL = "https://api.perplexity.ai/chat/completions"

def perplexity_chat(prompt, system_prompt=None, temperature=0.7):
    headers = {
        "Authorization": f"Bearer {PPLX_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    data = {
        "model": "sonar-pro",
        "messages": messages,
        "temperature": temperature
    }

    response = requests.post(PPLX_API_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content'].strip()

def detect_intent_and_extract(user_input):
    """
    Detect user intent and extract a clean query using Perplexity AI.
    Falls back to rule-based detection only if API fails.
    """
    try:
        result = detect_intent_and_extract_pplx(user_input)
        if result and result.get("intent"):
            return result
    except Exception as e:
        print("❌ Perplexity intent fallback error:", e)

    input_lower = user_input.strip().lower()
    file_keywords = ["file", "document", "report", "sheet", "policy"]
    low_context_phrases = [
        "hi", "hello", "how are you", "thank you", "what can you do",
        "who are you", "good morning", "good evening", "hey", "help"
    ]

    for phrase in low_context_phrases:
        if phrase in input_lower:
            return {"intent": "general_response", "data": user_input}

    for kw in file_keywords:
        if kw in input_lower:
            return {
                "intent": "file_search_prompt",
                "data": ""
            }

    return {"intent": "general_response", "data": user_input}

def detect_intent_and_extract_pplx(user_input):
    """
    Use Perplexity AI to classify user intent and optionally extract a clean query.
    Returns a strict JSON object with one of these intents:
    - HR_Admin
    - file_search
    - file_search_prompt
    - general_response
    """
    system_prompt = (
        "You're an intent classifier for an HR + file assistant chatbot. Classify the user input into one of these intents:\n\n"
        "- HR_Admin: For queries about HR policies, leave, NID, ID numbers, benefits, holidays, onboarding, payroll, work schedule, HR rules, or employee matters.\n"
        "- file_search: If the user is clearly asking to search, find, retrieve, preview, or get a specific document or file (e.g., 'valuation report', 'pike 2023 financials', 'supernova deck').\n"
        "- file_search_prompt: If the user gives a vague phrase like 'I need a file', 'I'm looking for something', 'Can you help me find a document?', without specifying what.\n"
        "- general_response: For greetings, chitchat, jokes, thank you, or unrelated input.\n\n"
        "Respond strictly in this JSON format:\n"
        "{\"intent\": \"intent_name\", \"data\": \"cleaned relevant keyword(s) or query\"}\n\n"
        "Examples:\n"
        "User: What is the maternity leave policy?\n"
        "\u2192 {\"intent\": \"HR_Admin\", \"data\": \"maternity leave policy\"}\n\n"
        "User: Show me the 2024 financial report file\n"
        "\u2192 {\"intent\": \"file_search\", \"data\": \"2024 financial report\"}\n\n"
        "User: I am looking for a file\n"
        "\u2192 {\"intent\": \"file_search_prompt\", \"data\": \"\"}\n\n"
        "Strict rules:\n"
        "- Classify vague inputs (e.g., 'I need a document', 'show me a file', 'I'm looking for something') as 'file_search_prompt'.\n"
        "- Do NOT classify inputs as 'file_search' if they only mention 'file', 'document', etc. without a topic or description.\n"
        "- Remove suffix words like 'file', 'document','excel', 'sheet','list','pdf','docx','txt' from the extracted data.\n"
        "- Classify questions like '2023 pike valuation', 'Q4 deck', 'supernova update', 'board report' as file_search.\n"
        "- Output must be strict valid JSON with no extra commentary.\n\n"
        f"User input:\n{user_input}"
    )

    try:
        response = perplexity_chat(user_input, system_prompt=system_prompt, temperature=0.2)
        result = json.loads(response)

        if "data" in result and isinstance(result["data"], str):
            result["data"] = re.sub(r"\\b(file|document|sheet|pdf|docx|txt)\\b", "", result["data"], flags=re.IGNORECASE).strip()

        return result
    except Exception as e:
        print("❌ Perplexity error during intent detection:", e)
        return {"intent": "general_response", "data": ""}

def answer_general_query(user_input):
    """
    Handles general queries like greetings or casual small talk using Perplexity, with short replies.
    """
    try:
        return perplexity_chat(
            user_input,
            system_prompt="You are a helpful assistant named ECHO. Reply clearly and briefly to casual user messages questions like 'what can you do?' or 'who are you?'. Reply naturally, warmly, and briefly. Do not include source reference numbers like [1], [2], etc. in your response. If the message is a greeting like 'hi', 'good morning', 'good afternoon', just return a simple, friendly 1-line greeting like chatGPT greeting response. Do not add suggestions or information.",
            temperature=0.4
        )
    except Exception as e:
        print("❌ Error in dynamic general response:", e)
        return "Hi there!"


def answer_with_chat_style(user_input):
    """
    Handles broad questions with world or knowledge-based tone.
    """
    try:
        system_prompt = (
            "You are ChatGPT, an intelligent assistant that can answer general world knowledge, "
            "recent events, news-style questions, and everyday queries. "
            "Even if some events are recent, do your best to provide an informed response."
        )
        return perplexity_chat(user_input, system_prompt=system_prompt, temperature=0.7)
    except Exception as e:
        print("❌ Error in Perplexity-style fallback:", e)
        return "⚠️ I'm having trouble providing that answer right now."
