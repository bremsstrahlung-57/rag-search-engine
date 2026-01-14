import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
model = os.environ.get("MODEL", "gemini-2.5-flash")

client = genai.Client(api_key=api_key)


def llm_query_enhance(query: str, method: str) -> str:
    match method:
        case "spell":
            return spell_correction(query, method)
        case "rewrite":
            return rewrite_query(query, method)
        case "expand":
            return expand_query(query, method)


def spell_correction(query: str, method: str) -> str:
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


def rewrite_query(query: str, method: str) -> str:
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


def expand_query(query: str, method: str) -> str:
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
