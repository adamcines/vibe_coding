import asyncio
import json
import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Reeds Jobs API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GREENHOUSE_BOARDS = [
    "riskified",
    "fireblocks",
    "pagayais",
    "gongio",
    "lightricks",
    "similarweb",
    "melio",
    "wizinc",
    "yotpo",
    "catonetworks",
]
GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)


async def fetch_board(client: httpx.AsyncClient, token: str) -> list[dict]:
    """Fetch all jobs for a single Greenhouse board and tag them with the company."""
    response = await client.get(GREENHOUSE_URL.format(token=token))
    response.raise_for_status()
    data = response.json()
    jobs = []
    for job in data.get("jobs", []):
        location = job.get("location") or {}
        jobs.append(
            {
                "title": job.get("title"),
                "location": location.get("name"),
                "apply_url": job.get("absolute_url"),
                "company": token,
            }
        )
    return jobs


async def collect_jobs() -> list[dict]:
    """Fetch jobs from all configured Greenhouse boards concurrently and combine them."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        results = await asyncio.gather(
            *(fetch_board(client, token) for token in GREENHOUSE_BOARDS),
            return_exceptions=True,
        )

    jobs: list[dict] = []
    for token, result in zip(GREENHOUSE_BOARDS, results):
        if isinstance(result, Exception):
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch jobs for board '{token}': {result}",
            )
        jobs.extend(result)

    return jobs


@app.get("/jobs")
async def get_jobs() -> dict:
    """Fetch and combine jobs from all configured Greenhouse boards."""
    jobs = await collect_jobs()
    return {"count": len(jobs), "jobs": jobs}


class RankRequest(BaseModel):
    cv: str
    role: str
    top_k: int = 10


RANK_SCHEMA = {
    "type": "object",
    "properties": {
        "ranked": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "score": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "score", "reason"],
            },
        }
    },
    "required": ["ranked"],
}


async def rank_with_gemini(cv: str, role: str, jobs: list[dict], top_k: int) -> list[dict]:
    """Ask Gemini to score each job against the CV + desired role and return the best matches."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    listing = "\n".join(
        f"{i}. {job.get('title')} | {job.get('company')} | {job.get('location')}"
        for i, job in enumerate(jobs)
    )
    prompt = (
        "You are a career-matching assistant. Score how well each job posting fits the "
        "candidate, on a 0-100 scale, based on their CV and the role they want. Weigh the "
        "desired role heavily, then transferable skills and experience from the CV.\n\n"
        f"DESIRED ROLE:\n{role}\n\n"
        f"CANDIDATE CV:\n{cv}\n\n"
        f"JOBS (index. title | company | location):\n{listing}\n\n"
        f"Return only the {top_k} best-fitting jobs, sorted by score descending. For each, give "
        "its index, an integer score (0-100), and a reason of at most 25 words explaining the fit."
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": RANK_SCHEMA,
        },
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            GEMINI_URL, params={"key": api_key}, json=payload
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini request failed ({response.status_code}): {response.text[:300]}",
        )

    try:
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        ranked = json.loads(text)["ranked"]
    except (KeyError, IndexError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Could not parse Gemini response: {exc}")

    results = []
    for item in ranked:
        idx = item.get("index")
        if not isinstance(idx, int) or not 0 <= idx < len(jobs):
            continue
        results.append({**jobs[idx], "score": item.get("score"), "reason": item.get("reason")})
    results.sort(key=lambda r: r.get("score") or 0, reverse=True)
    return results


@app.post("/rank")
async def rank_jobs(request: RankRequest) -> dict:
    """Rank all fetched jobs against a CV and desired role, returning scored matches."""
    jobs = await collect_jobs()
    top_k = max(1, min(request.top_k, len(jobs)))
    ranked = await rank_with_gemini(request.cv, request.role, jobs, top_k)
    return {
        "role": request.role,
        "considered": len(jobs),
        "returned": len(ranked),
        "results": ranked,
    }
