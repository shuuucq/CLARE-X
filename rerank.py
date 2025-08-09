import asyncio
import aiohttp
import json
import re
import logging
from tqdm.asyncio import tqdm_asyncio


logging.basicConfig(level=logging.INFO)

OLLAMA_API_URL = ""
CONCURRENCY_LIMIT = 1
MAX_RETRIES = 5
REQUEST_TIMEOUT = 100
BATCH_DELAY = 0.2

# =====================
# =====================
async def call_ollama_for_rerank(session, prompt, system_prompt, semaphore):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens":4096
    }

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.post(OLLAMA_API_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result["choices"][0]["message"]["content"], True
                    else:
                        text = await resp.text()
                        logging.warning(f"HTTP {resp.status} (attempt {attempt}): {text}")
            except asyncio.TimeoutError:
                logging.warning(f"Timeout on attempt {attempt}")
            except Exception as e:
                logging.warning(f"Request error on attempt {attempt}: {e}")
            await asyncio.sleep(1.2 * attempt + BATCH_DELAY)

    return "Failed after max retries", False

import re

def extract_and_convert_ranked_response(content: str, user_id: str):

    content = re.sub(r"</?.*?>", "", content).strip()
    lines = content.splitlines()
    results = []

    for line in lines:
        line = line.strip()
        
        match_with_reason = re.match(r"(\d+)\.\s*Reviewer\s+(\d+):\s*(.+)", line)
        if match_with_reason:
            rank = int(match_with_reason.group(1))
            reviewer_id = match_with_reason.group(2)
            reason = match_with_reason.group(3).strip()
            results.append((reviewer_id, rank, reason))
            continue

        match_no_reason = re.match(r"(\d+)\.\s*Reviewer\s+(\d+)", line)
        if match_no_reason:
            rank = int(match_no_reason.group(1))
            reviewer_id = match_no_reason.group(2)
            reason = ""
            results.append((reviewer_id, rank, reason))
            continue

    if not results:
        raise ValueError("No valid ranked reviewer list found in response")

    sorted_results = sorted(results, key=lambda x: x[1])
    return sorted_results



# =====================
# =====================
def build_prompt(submission, candidates):
    input_data = {
        "submission": {
            "title": submission.get("title", ""),
            "abstract": submission.get("abstract", "")
        },
        "candidates": [
            {
                "reviewer_id": r.get("reviewer_id", ""),
                "scores": r.get("scores", ""),
                "specialty": r.get("specialty", []),
                "user_profile": r.get("user_profile", "")
            }
            for r in candidates
        ]
    }

    return (
        "You will be given information about a submission and 20 reviewer candidates.\n"
        "Each reviewer includes reviewer_id, model score, specialty, and profile.\n"
        "Please rank all reviewers from 1 to 20 based on their suitability for the submission.\n"
        "For the top 10 reviewers only, write a short explanation (max 50 words) why they are a good match.\n"
        "For reviewers ranked 11 to 20, just list the reviewer ID without explanation.\n"
        "Output format must be like this:\n"
        "1. Reviewer <id>: <reason>\n"
        "2. Reviewer <id>: <reason>\n"
        "...\n"
        "11. Reviewer <id>\n"
        "...\n"
        "Here is the input:\n"
        + json.dumps(input_data, ensure_ascii=False, indent=2)
        + "\n/no_think"
    )


# =====================
# =====================
async def run_rerank_with_gpt_batch(input_samples):
    rerank_results = {}
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    system_prompt = "You are an expert editor assistant helping match reviewers to submissions."

    async with aiohttp.ClientSession() as session:
        async def process_one(sample):
            user_id = sample["user_id"]
            prompt = build_prompt(sample["submission"], sample["candidates"])
            content, success = await call_ollama_for_rerank(session, prompt, system_prompt, semaphore)
            # print(" content:", content)

            if not success:
                logging.error(f"Rerank failed for {user_id}: {content}")
                rerank_results[user_id] = {
                    "status": "error",
                    "message": content,
                    "results": []
                }
                return

            try:
                parsed = extract_and_convert_ranked_response(content,user_id)
                rerank_results[user_id] = parsed
            except Exception as e:
                logging.error(f"Parsing error for {user_id}: {e}")
                rerank_results[user_id] = {
                    "status": "error",
                    "message": str(e),
                    "results": []
                }

        tasks = [process_one(sample) for sample in input_samples]
        await tqdm_asyncio.gather(*tasks, desc="Reranking")

    return rerank_results
