import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        page.on("console", lambda msg: print(f"Console: {msg.text}"))
        page.on("pageerror", lambda exc: print(f"Error: {exc}"))
        
        await page.goto("file:///home/kim/Projects/mir/plots/feature_explorer.html")
        await asyncio.sleep(2)  # Wait for load
        # Switch to radar
        await page.evaluate("setMode('radar')")
        await asyncio.sleep(1)
        await browser.close()

asyncio.run(main())
