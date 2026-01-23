from lib.llm_funcs import client,model
import mimetypes
from google.genai import types


def read_img(query, img):
    system_prompt="""Given the included image and text query,
    rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary"""

    with open(img, "rb") as f:
        img_bytes = f.read()

    mime, _ = mimetypes.guess_type(img)
    mime = mime or "image/jpeg"

    parts = [
        system_prompt,
        types.Part.from_bytes(data=img_bytes,mime_type=mime),
        query.strip(),
    ]

    response = client.models.generate_content(
        model=model,
        contents=parts,
    )

    print(f"Rewritten query: {response.text}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")

    return response.text
