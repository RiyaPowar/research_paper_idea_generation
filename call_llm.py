from groq import Groq
import os
import dotenv
from load_artifacts import retrieve, format_context

dotenv.load_dotenv()

from prompt import PROMPT_TEMPLATE

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_ideas(topic):
    retrieved_docs = retrieve(topic)
    context = format_context(retrieved_docs)

    prompt = PROMPT_TEMPLATE.format(topic=topic, context=context)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )

    return response.choices[0].message.content, retrieved_docs
