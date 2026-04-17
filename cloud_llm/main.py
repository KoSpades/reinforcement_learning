import os
import sys
from pathlib import Path

from google import genai
from groq import Groq


GEMINI_MODEL = "gemini-2.5-flash" # ["gemini-2.5-flash"]
GROQ_MODEL = "openai/gpt-oss-120b" # ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"]
TEMPERATURE = 0.1


def load_env_file(path=None):
    if path is None:
        path = Path(__file__).with_name(".env")

    if not os.path.exists(path):
        return

    with open(path) as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip("\"'")


def require_env(name):
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Missing {name}. Add it to cloud_llm/.env, then run python cloud_llm/main.py again.")
    return value


def ask_gemini(prompt):
    client = genai.Client(api_key=require_env("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": TEMPERATURE,
        },
    )
    return response.text


def ask_groq(prompt):
    client = Groq(api_key=require_env("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=2048,
    )
    return response.choices[0].message.content


if __name__ == "__main__":

    PROVIDER = "gemini"  # one of "groq" or "gemini"
    load_env_file()

    PROMPT_1 = (
        "How many 5-digit numbers can be formed using the digits 1,2,3,4,5,6,7 "
        "with no repetition, such that the number is even and the digits are in "
        "strictly increasing order from left to right?"
    )

    PROMPT_2 = (
        "How many distinct arrangements of the letters in MISSISSIPPI have no two S's adjacent and no two P's adjacent?"
    )

    if PROVIDER == "gemini":
        answer = ask_gemini(PROMPT_2)
    elif PROVIDER == "groq":
        answer = ask_groq(PROMPT_2)
    else:
        sys.exit('PROVIDER must be either "gemini" or "groq".')

    print(answer)
