import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
from typing import List
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool, set_tracing_disabled, OpenAIChatCompletionsModel
from myagents.collectoragent import NewsItem

# === ENV & DB setup ===
load_dotenv()
set_tracing_disabled(True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is missing in .env")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL is missing in .env")

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# === OpenAI client & model ===
client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

# === Pydantic model for structured tags ===
class TaggedNewsItem(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    @classmethod
    def from_agent_text(cls, text: str) -> List["TaggedNewsItem"]:
        """
        Convert JSON text output from agent into list of TaggedNewsItem
        """
        import json
        try:
            data = json.loads(text)
            return [cls(symbols=item.get("symbols", []), tags=item.get("tags", [])) for item in data]
        except Exception:
            return []

# === Fetch untagged news from DB ===
async def fetch_untagged_news(max_items: int = 10) -> List[NewsItem]:
    async with async_session() as session:
        result = await session.execute(
            select(NewsItem).where((NewsItem.tags == []) | (NewsItem.symbols == [])).limit(max_items)
        )
        return result.scalars().all()

# === Tool: Tag news using OpenAI agent ===
@function_tool
async def tag_news_tool(news_text: str) -> List[TaggedNewsItem]:
    """
    Input: A string containing multiple news items, numbered.
    Output: List of TaggedNewsItem with symbols and tags.
    """
    prompt = f"""
You are a financial tagging assistant.
Respond ONLY with valid JSON — no explanations.

For each news item below (title + summary):
1. Extract stock symbols (e.g., AAPL, TSLA, MSFT)
2. Extract relevant tags such as: earnings, macro, fed, AI, tech, energy, crypto, etc.
3. Output a JSON array in the same order of news items.

News items:
{news_text}
"""
    try:
        resp = await client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0
        )
        text = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ AI tagging failed: {e}")
        return []

    return TaggedNewsItem.from_agent_text(text)

# === Tagger Agent ===
tagger_agent = Agent(
    name="NewsTagger",
    instructions="""
You generate structured tags for financial news.
- Use `tag_news_tool` to extract symbols and relevant tags for each news item.
- Output must be JSON, order should match the input.
- Focus on accurate, relevant financial and market tags.
""",
    tools=[tag_news_tool],
    model=model
)

# === Update DB with tags ===
async def update_tags(items: List[NewsItem], tagged_items: List[TaggedNewsItem]):
    async with async_session() as session:
        for item_obj, tag_obj in zip(items, tagged_items):
            session.add(
                update(NewsItem)
                .where(NewsItem.id == item_obj.id)
                .values(symbols=tag_obj.symbols, tags=tag_obj.tags)
            )
        await session.commit()

# === Main workflow ===
async def main():
    items = await fetch_untagged_news(max_items=10)
    if not items:
        print("No untagged news to process.")
        return

    # Convert news to string for agent
    news_text = "\n\n".join(f"{idx+1}. {n.title}\n{n.summary or ''}" for idx, n in enumerate(items))

    # Run tagger agent
    tagged_result = await Runner.run(tagger_agent, news_text)
    raw_output = tagged_result.final_output if tagged_result else ""

    if not raw_output:
        print("No tags generated.")
        return

    # Convert agent output into structured objects
    tagged_items = TaggedNewsItem.from_agent_text(raw_output)

    # Update DB
    await update_tags(items, tagged_items)

    # Print nicely
    for i, t in zip(items, tagged_items):
        print(f"{i.title}\n  Symbols: {t.symbols}\n  Tags: {t.tags}\n")

    print(f"✅ Tagged {len(items)} news items and saved to DB.")

# === Wrapper for external calls ===
async def run_tagger(items=None):
    if not items:
        items = await fetch_untagged_news(max_items=10)
    if not items:
        print("No untagged news to process.")
        return []

    news_text = "\n\n".join(f"{idx+1}. {n.title}\n{n.summary or ''}" for idx, n in enumerate(items))
    tagged_result = await Runner.run(tagger_agent, news_text)
    raw_output = tagged_result.final_output if tagged_result else ""
    tagged_items = TaggedNewsItem.from_agent_text(raw_output)

    if tagged_items:
        await update_tags(items, tagged_items)
        for i, t in zip(items, tagged_items):
            print(f"{i.title}\n  Symbols: {t.symbols}\n  Tags: {t.tags}\n")
        print(f"✅ Tagged {len(items)} news items and saved to DB.")

    return tagged_items

if __name__ == "__main__":
    asyncio.run(main())
