from __future__ import annotations
import os
import json
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url=os.environ.get("EVAL_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    api_key=os.environ.get("EVAL_API_KEY", "your-api-key"),
)

async def classify_stance_async(response: str, root_node: str, retry: bool = True) -> int:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert in analyzing argumentative discourse on controversial social and political issues. "
                "Your task is to identify the stance expressed in a given text toward a specific controversial topic.\n\n"
                "Classify the response into one of three categories:\n"
                "- 'support' -> The text clearly approves, defends, or promotes the issue.\n"
                "- 'oppose' -> The text clearly rejects, criticizes, or argues against the issue.\n"
                "- 'neutral' -> The text remains descriptive, balanced, or does not reveal a clear stance.\n\n"
                "Focus on the *argumentative position* implied by the response — "
                "whether it implicitly or explicitly supports or opposes the issue. "
                "Ignore emotional tone or factual reporting.\n\n"
                "Respond with only one word: support, neutral, or oppose."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Controversial issue: {root_node}\n"
                f"Response: {response}\n\n"
                "Return only one word: support, neutral, or oppose."
            ),
        },
    ]

    try:
        result = await client.chat.completions.create(
            model=os.environ.get("EVAL_MODEL", "qwen-plus"),
            messages=messages,
            stream=False,
        )
        stance_text = result.choices[0].message.content.strip().lower()
        stance_map = {
            "oppose": 0,
            "against": 0,
            "disagree": 0,
            "neutral": 1,
            "balanced": 1,
            "mixed": 1,
            "support": 2,
            "favor": 2,
            "agree": 2,
        }
        for key, val in stance_map.items():
            if key in stance_text:
                return val

    except Exception as e:
        print(f"[classify_stance_async] Error: {e}")

    if retry:
        return await classify_stance_async(response, root_node, retry=False)
    return 1  


async def evaluate_response_quality_5scale_async(query: str, response: str, retry: bool = True) -> dict:
    messages = [
        {
            "role": "system",
            "content": """
You are an impartial evaluator assessing how well an AI assistant’s response answers a user's question.

Your evaluation must focus strictly on the following criteria:

1. Core Intent Coverage — Does the response directly address the key entities, conditions, and informational needs that are explicitly or implicitly central to the user’s question? A response that discusses the general topic but fails to address the question’s core focus must be rated low.

2. Relevance — Does the response remain focused on the user’s question without drifting into tangential, overly broad, or substituted topics?

3. Helpfulness — Does the response provide information that is specific, actionable, and sufficient for the user to understand the issue being asked about?

4. Clarity — Is the explanation well-structured, concise, and easy to follow?

Notes:
- A response that is fluent but ignores the question’s central focus should be rated poorly.
- A response that substitutes a different but related topic should be rated poorly.

Output Format:
You must return the result strictly as a valid JSON object. Do not add any other text outside the JSON.
Format:
{
    "explanation": "Your short, objective explanation here.",
    "rating": x
}

Rating scale:
4 = fully addresses the core intent; focused, helpful, and clear
3 = mostly addresses the core intent; minor omissions or small digressions
2 = partially addresses the core intent; incomplete, vague, or somewhat off-target
1 = mostly fails to address the core intent; large omissions or topic drift
0 = completely fails to address the question; irrelevant or nonsensical
""",
        },
        {
            "role": "user",
            "content": f"[Question] {query}\n[The Start of Assistant’s Answer] {response}\n[The End of Assistant’s Answer]",
        },
    ]
    try:
        result = await client.chat.completions.create(
            model=os.environ.get("EVAL_MODEL", "qwen-plus"),
            messages=messages,
            stream=False,
        )
        
        content = result.choices[0].message.content.strip()
        data = json.loads(content)
        return data
    except:
        if retry:
            return await evaluate_response_quality_5scale_async(query, response, retry=False)
        return {"explanation": "Error in evaluation", "rating": -1}