import os
import asyncio
from playwright.async_api import async_playwright
from sentence_transformers import SentenceTransformer

NEWS_TXT_PATH = "news_knowledge.txt"
CHUNK_SIZE = 500  

embedder = SentenceTransformer("all-MiniLM-L6-v2")

async def scrape_wktv_news():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://www.wktv.com/news/")

        article_links = await page.eval_on_selector_all(
            ".col-md-8.fixed-col-left.left-col a[href*='/article_']",
            "els => [...new Set(els.map(e => e.href))]"
        )

        article_data = []
        for url in article_links:
            try:
                await page.goto(url)
                await page.wait_for_selector("article", timeout=5000)
                title = await page.title()
                paragraphs = await page.eval_on_selector_all(
                    "article p", "els => els.map(e => e.innerText)"
                )
                full_text = title + "\n" + "\n".join(paragraphs)
                article_data.append({"title": title, "content": full_text})
                print(f"✅ Collected: {title}")
            except Exception as e:
                print(f"❌ Failed to scrape {url}: {e}")

        await browser.close()
        return article_data

def save_articles_to_file(articles):
    with open(NEWS_TXT_PATH, "w", encoding="utf-8") as f:
        for idx, article in enumerate(articles):
            f.write(f"ARTICLE {idx}\n")
            f.write(f"Title: {article['title']}\n")
            f.write(f"Content: {article['content']}\n")
            f.write("=" * 50 + "\n")

def load_article_by_index(index):
    """Load a single article from the file by its index."""
    with open(NEWS_TXT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_article = []
    article_started = False
    for line in lines:
        if line.startswith(f"ARTICLE {index}"):  # Find the correct article
            article_started = True
        elif article_started:
            if line.startswith("ARTICLE"):  # Next article, stop reading
                break
            current_article.append(line.strip())

    return "\n".join(current_article) if current_article else None

async def main():
    print("📰 Scraping WKTV News...")
    articles = await scrape_wktv_news()
    if articles:
        save_articles_to_file(articles)
        print(f"✅ {len(articles)} articles saved to news_knowledge.txt.")
    else:
        print("⚠️ No articles scraped.")

if __name__ == "__main__":
    asyncio.run(main())
