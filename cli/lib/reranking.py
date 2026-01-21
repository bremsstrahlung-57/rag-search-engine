import os
import re

from dotenv import load_dotenv
from google import genai
from groq import Groq

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
model = os.environ.get("MODEL", "gemini-2.5-flash")
groq_api_key = os.environ.get("GROQ_API_KEY")
groq_model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

client = genai.Client(api_key=api_key)
groq_client = Groq(api_key=groq_api_key)


def llm_query_enhance(query: str, method: str) -> str | None:
    match method:
        case "spell":
            return spell_correction(query, method)
        case "rewrite":
            return rewrite_query(query, method)
        case "expand":
            return expand_query(query, method)
        case _:
            return query


def spell_correction(query: str, method: str) -> str | None:
    response = client.models.generate_content(
        model=model,
        contents=f"""Fix any spelling errors in this movie search query.
        Only correct obvious typos. Don't change correctly spelled words.
        Query: "{query}"
        If no errors, return the original query.
        Corrected:""",
    )
    print(f"Enhanced query ({method}): '{query}' -> '{response.text}'\n")
    return response.text


def rewrite_query(query: str, method: str) -> str | None:
    response = client.models.generate_content(
        model=model,
        contents=f"""Rewrite this movie search query to be more specific and searchable.

        Original: "{query}"

        Consider:
        - Common movie knowledge (famous actors, popular films)
        - Genre conventions (horror = scary, animation = cartoon)
        - Keep it concise (under 10 words)
        - It should be a google style search query that's very specific
        - Don't use boolean logic

        Examples:

        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

        Rewritten query:""",
    )
    print(f"Enhanced query ({method}): '{query}' -> '{response.text}'\n")
    return response.text


def expand_query(query: str, method: str) -> str | None:
    response = client.models.generate_content(
        model=model,
        contents=f"""Expand this movie search query with related terms.

        Add synonyms and related concepts that might appear in movie descriptions.
        Keep expansions relevant and focused.
        This will be appended to the original query.

        Examples:

        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
        - "action movie with bear" -> "action thriller bear chase fight adventure"
        - "comedy with bear" -> "comedy funny bear humor lighthearted"

        Query: "{query}"
        """,
    )
    print(f"Enhanced query ({method}): '{query}' -> '{response.text}'\n")
    return response.text


def llm_reranking(query, title, desc):
    response = client.models.generate_content(
        model=model,
        contents=f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {title} - {desc}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Give me ONLY the number in your response, no other text or explanation.

        Score:""",
    )

    text = (response.text or "").strip()

    match = re.search(r"\b(10|[0-9])\b", text)
    if match:
        return int(match.group(1))
    return 0


def groq_reranking(query, title, desc, rrf_rank, sem_rank, bm_rank):
    response = groq_client.chat.completions.create(
        model=groq_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a deterministic relevance scoring function. "
                    "Your task is to output exactly one integer from 0 to 10. "
                    "Do not explain your reasoning."
                ),
            },
            {
                "role": "user",
                "content": f"""
                Score how relevant the following movie is to the search query.

                Definitions:
                    - 0 = completely irrelevant
                    - 5 = somewhat relevant
                    - 10 = perfect match

                Query: "{query}"

                Movie: {title} - {desc}

                Retrieval signals (lower rank = better):
                    - RRF rank: {rrf_rank}
                    - Semantic rank: {sem_rank}
                    - BM25 rank: {bm_rank}

                Rules:
                    - Use the description and query as the primary signal.
                    - Use the ranks only as weak supporting signals.
                    - Output exactly one integer between 0 and 10.

                Score:""",
            },
        ],
    )

    text = (response.choices[0].message.content or "").strip()
    score = int(text) if text.isdigit() and 0 <= int(text) <= 10 else 0

    return score


def batch_reranking(query, doc_list_str):
    response = client.models.generate_content(
        model=model,
        contents=f"""Rank these movies by relevance to the search query.

        Query: "{query}"

        Movies:
        {doc_list_str}

        Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

        [75, 12, 34, 2, 1]
        """,
    )

    return response.text


def evaluate_result(query, results):
    response = client.models.generate_content(
        model=model,
        contents=f"""Rate how relevant each result is to this query on a 0-3 scale:

        Query: "{query}"

        Results:
        {results}

        Scale:
        - 3: Highly relevant
        - 2: Relevant
        - 1: Marginally relevant
        - 0: Not relevant

        Do NOT give any numbers out than 0, 1, 2, or 3.

        Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

        [2, 0, 3, 2, 0, 1]""",
    )
    return response.text
