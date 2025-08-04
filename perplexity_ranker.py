import os
import requests
from dotenv import load_dotenv

load_dotenv()

PPLX_API_KEY = os.getenv("PPLX_API_KEY")
PPLX_API_URL = "https://api.perplexity.ai/chat/completions"

def rank_files_with_perplexity(query, files):
    print(query)
    headers = {
        "Authorization": f"Bearer {PPLX_API_KEY}",
        "Content-Type": "application/json"
    }

    # Construct text summary for Perplexity input
    file_descriptions = "\n".join(
        f"{i+1}. {f['name']}\n{f.get('extracted_text', '')[:1000]}"
        for i, f in enumerate(files)
    )

    system_prompt = (
        "You are a document assistant AI. Your job is to rank the following files "
        "in order of how closely each file matches the user's query.\n\n"
        "Instructions:\n"
        "- Focus on exact relevance to the query.\n"
        "- Use both the file name and content to determine accuracy.\n"
        "- Do NOT infer recency, importance, or guess missing context.\n"
        "- Match based on clear textual similarity — highest match to query goes first.\n"
        "- Respond ONLY in this format:\n"
        "Ranked files:\n1. filename\n2. filename\n..."
    )

    user_prompt = (
        f"User query: {query}\n\n"
        f"Files:\n{file_descriptions}\n\n"
        "Rank these files from most to least relevant based strictly on the query."
    )

    data = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    response = requests.post(PPLX_API_URL, headers=headers, json=data)
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]

    # Parse filenames from the output
    ordered_names = []
    for line in content.splitlines():
        line = line.strip()
        if line and any(c.isdigit() for c in line):
            parts = line.split('.', 1)
            if len(parts) > 1:
                name = parts[1].strip()
                ordered_names.append(name)

    # Return files in the ranked order
    ranked = []
    for name in ordered_names:
        match = next((f for f in files if f["name"] == name), None)
        if match and match not in ranked:
            ranked.append(match)

    return ranked


def call_perplexity_chat(prompt, system="You are a helpful assistant."):
    headers = {
        "Authorization": f"Bearer {PPLX_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    }

    res = requests.post(PPLX_API_URL, headers=headers, json=data)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]