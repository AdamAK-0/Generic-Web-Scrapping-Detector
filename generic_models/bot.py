import os
import asyncio
import random

os.environ["BEZIER_NO_EXTENSION"] = "True"
os.environ["BEZIER_IGNORE_VERSION_CHECK"] = "True"

from patchright.async_api import async_playwright
from python_ghost_cursor.playwright_async import create_cursor

# Your Graph Data (Edges)
EDGES = [
    ("/atlas_shop/", ["/catalog", "/deals", "/about", "/search", "/cart"]),
    ("/catalog", ["/catalog/water", "/catalog/sensors", "/catalog/kits", "/catalog/field", "/search", "/atlas_shop/"]),
    ("/catalog/water", ["/p/aqua", "/p/river", "/p/filter", "/catalog", "/cart"]),
    ("/catalog/sensors", ["/p/flow", "/p/node", "/p/camera", "/catalog", "/cart"]),
    ("/catalog/kits", ["/p/reservoir", "/p/starter", "/p/pro", "/catalog", "/deals"]),
    ("/catalog/field", ["/p/boots", "/p/pack", "/p/meter", "/catalog"]),
    ("/deals", ["/p/starter", "/p/filter", "/catalog", "/cart"]),
    ("/search", ["/catalog/water", "/catalog/sensors", "/p/aqua", "/p/flow", "/atlas_shop/"]),
    ("/about", ["/atlas_shop/", "/catalog", "/contact"]),
    ("/contact", ["/atlas_shop/", "/about"]),
    ("/cart", ["/catalog", "/deals", "/atlas_shop/"]),
    # Product pages mostly loop back to Catalog, Cart, or Search
    ("/p/aqua", ["/catalog", "/cart", "/search"]),
    ("/p/river", ["/catalog", "/cart", "/search"]),
    ("/p/filter", ["/catalog", "/cart", "/search"]),
]

EDGE_MAP = dict(EDGES)

async def navigate(page, cursor, base_url, current_path, depth=0, max_depth=10):
    if depth >= max_depth:
        return

    full_url = f"{base_url.rstrip('/')}{current_path}"
    print(f"[Depth {depth}] Visiting: {full_url}")
    
    try:
        await page.goto(full_url, wait_until="networkidle")
        
        # Human-like behavioral noise
        await cursor.move_to({"x": random.randint(100, 800), "y": random.randint(100, 600)})
        if random.random() > 0.5:
            await page.mouse.wheel(0, random.randint(300, 700))
        
        await asyncio.sleep(random.uniform(2, 5)) # Variable dwell time

        # Decide next node based on graph edges
        if current_path in EDGE_MAP:
            next_path = random.choice(EDGE_MAP[current_path])
            await navigate(page, cursor, base_url, next_path, depth + 1, max_depth)
            
    except Exception as e:
        print(f"Navigation error: {e}")

async def run_research_bot():
    base_url = "http://localhost:8061"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, channel="chrome")
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        
        page = await context.new_page()
        cursor = create_cursor(page)

        # Start at the Home Node
        await navigate(page, cursor, base_url, "/atlas_shop/")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(run_research_bot())
