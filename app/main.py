import os

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class Prompt(BaseModel):
    message: str


system_message = """
Your role is to generate a reference for the user's prompt. The user prompt will follow this object format:

{
    "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7231022/",
    "type": "journal",
    "dateVisited": "2024-05-05",
    "authors": "Dan M Livovsky, Teorora Pribic, Fernando Azpiroz, ",
    "publicationDate": "2020/04",
    "citationTitle": "Food, Eating, and the Gastrointestinal Tract",
    "citationJournalTitle": "Nutrients",
    "citationVolume": "12",
    "citationIssue": "4",
    "citationDoi": "10.3390/nu12040986"
}

As an example the above prompt would return the following reference:

"Livovsky, D. M., Pribic, T., & Azpiroz, F. (2020). Food, Eating, and the Gastrointestinal Tract. Nutrients, 12(4), 986. https://doi.org/10.3390/nu12040986. Visited 2024-05-05. Found on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7231022/"

The reference is based on APA referencing and you must ensure the following key points for creating and formatting citations and references:

General Principles:

Consistency: Ensure all references follow the same format, style, and punctuation.
Order: Organize reference lists alphabetically by the first author's last name.
Author Information:
List authors in the format "Last name, Initial(s)."
Use an ampersand (&) before the last author in a multi-author citation.
For more than 20 authors, list the first 19, then use an ellipsis (...) followed by the last author's name (no ampersand).
In-Text Citations:

Parenthetical Citations: Use the author-date format: (Author, Year). Include page numbers for direct quotes: (Author, Year, p. X).
Narrative Citations: Integrate the author’s name into the text, followed by the year in parentheses. Example: According to Smith (2020), ...
Reference List:

Books: Format as follows: Author, A. A. (Year). Title of the book. Publisher.
Journal Articles: Format as follows: Author, A. A. (Year). Title of the article. Title of the Journal, volume(issue), page numbers. https://doi.org/xx.xxx/yyyyyy (if available).
Websites: Format as follows: Author, A. A. (Year, Month Day). Title of the page/document. Website Name. URL.
Other Formats: Follow specific guidelines for conference papers, reports, dissertations, and other formats as outlined in the APA Publication Manual (7th edition).
Special Cases:

No Author: Use the title or the first few words of the title. Italicize book and periodical titles.
No Date: Use "n.d." in place of the year.
DOIs and URLs: Include DOIs when available. Use the full URL for online sources; omit "Retrieved from."

Lastly for journal articles specifically, you must include the date the article was visited and the URL from where it was found.

Only return the complete reference and nothing else based on the user's prompt.
"""

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
API_KEY = os.getenv('API_KEY')
api_key_header_scheme = APIKeyHeader(name="x-api-key")

app = FastAPI()


class CustomCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get('origin')
        if request.method == "OPTIONS":
            response = Response(status_code=204)
            response.headers["Access-Control-Allow-Origin"] = origin if origin and origin.startswith(
                "chrome-extension://") else "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            return response

        response = await call_next(request)
        if origin and origin.startswith("chrome-extension://"):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"
        return response


app.add_middleware(CustomCORSMiddleware)


@app.post("/generate-citation")
async def generate_citation(prompt: Prompt, key: str = Depends(api_key_header_scheme)):
    if not key in API_KEY:
        raise HTTPException(status_code=401, detail="Not authorized")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured.")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt.message}
            ],
            max_tokens=150,
            temperature=0.01,
        )
        return {"response": response.choices[0].message['content']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For debugging purposes
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
