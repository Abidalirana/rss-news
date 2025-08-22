from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, mapped_column, Mapped
from sqlalchemy import String, Integer, Text, DateTime, ARRAY, Boolean
from datetime import datetime
import logging
from sqlalchemy import select

#==================db=======================
import os
from dotenv import load_dotenv
load_dotenv()  # load .env in local dev, ignored in production on Render
DATABASE_URL = os.environ["DATABASE_URL"]
#=====================


engine = create_async_engine(DATABASE_URL, echo=False)  # Set echo=False to reduce logs
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

class NewsItem(Base):
    __tablename__ = "news_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(100), nullable=True)
    published_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=True)
    summary: Mapped[str] = mapped_column(Text, nullable=True)              # short key points
    detailed_summary: Mapped[str] = mapped_column(Text, nullable=True)     # <-- new column for full narrative
    tags: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])
    symbols: Mapped[list[str]] = mapped_column(ARRAY(String), default=[])
    url: Mapped[str] = mapped_column(String(500), unique=True, nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=True)
    publisher: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

def parse_datetime(dt):
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt
    try:
        return datetime.fromisoformat(dt)
    except Exception:
        return None

async def save_feed_items_to_db(items: list[dict]):
    try:
        async with async_session() as session:
            async with session.begin():
                urls = [item['url'] for item in items if item.get('url')]
                if not urls:
                    return

                result = await session.execute(
                    select(NewsItem.url).where(NewsItem.url.in_(urls))
                )
                existing_urls = set(row[0] for row in result.fetchall())

                for item in items:
                    url = item.get('url')
                    if not url or url in existing_urls:
                        continue
                    news_item = NewsItem(
                        title=item.get('title', 'No Title'),
                        source=item.get('source'),
                        published_at=parse_datetime(item.get('published_at')),
                        content=item.get('content'),
                        summary=item.get('summary'),
                        detailed_summary=item.get('detailed_summary'),  # <-- save detailed summary here
                        tags=item.get('tags', []),
                        symbols=item.get('symbols', []),
                        url=url,
                        provider=item.get('provider'),
                        publisher=False  # Always start unpublished
                    )
                    session.add(news_item)
            await session.commit()
    except Exception as e:
        logging.error(f"Failed to save feed items: {e}")
