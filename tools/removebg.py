import asyncio
import os
from pathlib import Path
from playwright.async_api import async_playwright

# adjust these as needed
INPUT_DIR = Path("data/images")
OUTPUT_DIR = Path("data/results")
CONCURRENCY = 3  # max number of simultaneous uploads/downloads
DOWNLOAD_TIMEOUT = 60_000  # ms

async def process_image(sem, context, image_path: Path):
    output_path = OUTPUT_DIR / image_path.name
    async with sem:
        page = await context.new_page()
        try:
            # 1. Go to removal.ai
            await page.goto("https://removal.ai")

            # 2. Upload
            await page.set_input_files('input.rm-upload-remove-background', str(image_path))

            # 3. Wait for the download link
            await page.wait_for_selector(
                'a.rm-btn.rm-btn-primary.rm-download-preview[href*="download"]',
                timeout=DOWNLOAD_TIMEOUT
            )

            # 4. Download
            async with page.expect_download() as download_info:
                await page.click('a.rm-btn.rm-btn-primary.rm-download-preview[href*="download"]')
            download = await download_info.value
            await download.save_as(str(output_path))

            print(f"✅ {image_path.name} → {output_path.name}")
        except Exception as e:
            print(f"❌ Failed {image_path.name}: {e}")
        finally:
            await page.close()

async def main():
    # ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # find all image files (jpg/png) in data/images
    image_paths = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
    if not image_paths:
        print(f"No images found in {INPUT_DIR}")
        return

    sem = asyncio.Semaphore(CONCURRENCY)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(accept_downloads=True)

        # schedule tasks
        tasks = [
            process_image(sem, context, img)
            for img in image_paths
        ]
        # run them (concurrently up to CONCURRENCY)
        await asyncio.gather(*tasks)

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
