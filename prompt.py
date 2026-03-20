PROMPT_TEMPLATE = """
You are an AI research assistant.

Use the following research papers as context:
{context}

Generate 3 novel research ideas in the domain of: {topic}

For each idea provide:

1. Title
2. Problem Statement
3. Proposed Method
4. Suggested Dataset
5. Evaluation Metrics
6. Possible Baseline Models
7. Why this idea is novel

Make it structured and technical.
Return output strictly in JSON format:

{{
  "ideas": [
    {{
      "title": "",
      "problem": "",
      "method": "",
      "dataset": "",
      "metrics": "",
      "baselines": "",
      "novelty": ""
    }}
  ]
}}
"""
