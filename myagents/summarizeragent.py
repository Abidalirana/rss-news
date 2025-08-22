import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
import re
from typing import List
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool, set_tracing_disabled, OpenAIChatCompletionsModel
from myagents.collectoragent import NewsItem, run_collector

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

# === Pydantic model for detailed summaries ===
class DetailedSummarizedNewsItem(BaseModel):
    number: int
    title: str
    key_points: List[str] = Field(..., min_items=5, max_items=8)
    detailed_text: str = Field(..., min_length=600)  # enforce MUCH longer summaries
    keywords: List[str] = Field(default_factory=list)

    @classmethod
    def from_agent_text(cls, text: str) -> List["DetailedSummarizedNewsItem"]:
        pattern = re.compile(
            r'(\d+)\.\s(.*?)\nKey points:\n((?:\s*•.*\n?)+)\n\n(.*?)\nKeywords:',
            re.DOTALL | re.MULTILINE
        )
        matches = pattern.findall(text)
        items = []
        for number, title, bullets, details in matches:
            points = [line.strip(" •") for line in bullets.strip().splitlines() if line.strip()]
            kw_match = re.search(rf'{re.escape(title)}.*?Keywords:\s*(.*)', text, re.DOTALL)
            keywords = [k.strip() for k in kw_match.group(1).split(",")] if kw_match else []
            items.append(cls(
                number=int(number),
                title=title,
                key_points=points,
                detailed_text=details.strip(),
                keywords=keywords
            ))
        return items

    def format_for_display(self) -> str:
        formatted = f"{self.number}. {self.title}\n"
        formatted += "\n".join(f"   • {pt}" for pt in self.key_points) + "\n"
        formatted += f"\n{self.detailed_text}\n"
        if self.keywords:
            formatted += f"\nKeywords: {', '.join(self.keywords)}\n"
        return formatted

# === Fetch unsummarized news from DB ===
async def fetch_unsummarized_news(max_items: int = 10) -> List[NewsItem]:
    async with async_session() as session:
        result = await session.execute(
            select(NewsItem).where((NewsItem.summary == None) | (NewsItem.summary == '')).limit(max_items)
        )
        return result.scalars().all()

# === Tool: Summarize news using OpenAI agent ===
@function_tool
async def summarize_news_tool(news_text: str) -> List[DetailedSummarizedNewsItem]:
    prompt = f"""
You are a professional financial news summarizer.

📌 STRICT RULES:
- Each summary must include 5–8 bullet points.
- The detailed narrative must be at least 25–30 lines long (minimum 4–5 full paragraphs).
- Structure the narrative into 3 clear sections (☁️ Market Impact, 🤖 Technology/Business, 💰 Financial Outlook).
- Each section MUST be 4–5 full sentences (minimum), written in professional tone.
- Use multiple paragraphs so it reads like a long-form market commentary.
- No fluff. Keep insights investor-focused (stocks, market strategy, deals, risks).
- Always include Keywords at the end.

Format the summary EXACTLY like this:

<STOCK HEADER>
GOOGL: Alphabet Stock Ticks Up as Google Strikes $10 Billion Cloud Deal with Meta
Aug 22, 2025, 13:58 GMT+5 — 2 min read

<TICKER SNAPSHOT>
GOOGL +0.22%
META +0.11%
AMZN -0.12%
MSFT -0.29%

<Key Points>
Key points:
• ...
• ...
• ...

<Detailed Narrative>
☁️ Market Impact  
4–5 sentences covering market sentiment, index movements, and sector reaction.

🤖 Technology/Business  
4–5 sentences explaining product launches, strategies, partnerships, or technology shifts.

💰 Financial Outlook  
4–5 sentences analyzing revenue, profit, valuations, risks, and investor expectations.

<Keywords>
Keywords: GOOGL, META, AI, Cloud, Earnings, Tech

News Articles:
{news_text}
"""
    try:
        resp = await client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=7000,   # allow long 30+ line answers
            temperature=0.3
        )
        text = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ AI summarization failed: {e}")
        return []

    return DetailedSummarizedNewsItem.from_agent_text(text)

# === News Summarizer Agent ===
news_agent = Agent(
    name="NewsSummarizer",
    instructions="""
You summarize financial news for readers and investors.
Always use `summarize_news_tool` to generate structured summaries.
✅ Each summary MUST have:
   - 5–8 bullet points
   - Detailed narrative ≥25 lines (ideally 30+)
   - 3 sections (Market Impact, Technology/Business, Financial Outlook)
   - At least 4 sentences per section
   - Keywords at the end
Keep numbering, avoid fluff.
""",
    tools=[summarize_news_tool],
    model=model
)

# === Update DB summaries ===
async def update_summaries(items: List[NewsItem], summarized_items: List[DetailedSummarizedNewsItem]):
    async with async_session() as session:
        for item_obj, summary_obj in zip(items, summarized_items):
            item_obj.summary = "\n".join(f"• {pt}" for pt in summary_obj.key_points)
            item_obj.detailed_summary = summary_obj.detailed_text
            item_obj.keywords = ", ".join(summary_obj.keywords)
            session.add(item_obj)
        await session.commit()

# === Main workflow ===
async def main():
    collected = await run_collector([{"symbols": ["AAPL", "TSLA"], "max_results": 5}])
    if not collected:
        print("Collector returned nothing, falling back to DB...")
        collected = await fetch_unsummarized_news(max_items=10)
    if not collected:
        print("No news to summarize.")
        return

    news_text = "\n\n".join(f"{idx+1}. {n.title}\n{n.summary or ''}" for idx, n in enumerate(collected))
    summarized_items_result = await Runner.run(news_agent, news_text)
    raw_output = summarized_items_result.final_output if summarized_items_result else ""
    if not raw_output:
        print("No summaries generated.")
        return

    summarized_items = DetailedSummarizedNewsItem.from_agent_text(raw_output)
    await update_summaries(collected, summarized_items)

    for s in summarized_items:
        print(s.format_for_display())

    print(f"✅ Summarized {len(collected)} items and saved to DB.")

# === Wrapper for external calls ===
async def run_summarizer(items=None):
    if not items:
        items = await fetch_unsummarized_news(max_items=10)
    if not items:
        print("No news to summarize.")
        return []

    news_text = "\n\n".join(f"{idx+1}. {n.title}\n{n.summary or ''}" for idx, n in enumerate(items))
    summarized_items_result = await Runner.run(news_agent, news_text)
    raw_output = summarized_items_result.final_output if summarized_items_result else ""
    summarized_items = DetailedSummarizedNewsItem.from_agent_text(raw_output)

    if summarized_items:
        await update_summaries(items, summarized_items)
        for s in summarized_items:
            print(s.format_for_display())
        print(f"✅ Summarized {len(items)} items and saved to DB.")
    return summarized_items

if __name__ == "__main__":
    asyncio.run(main())
