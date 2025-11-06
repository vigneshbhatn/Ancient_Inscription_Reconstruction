from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import requests
import os
import time
import base64
from urllib.parse import urljoin
import re


def scrape_github_pages_dropdown_content(url, folder='new_inscriptions'):
    """
    Scrape images by cycling through inscription dropdown options and downloading content images
    """
    os.makedirs(folder, exist_ok=True)

    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])

    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        print(f"Loading page: {url}")
        driver.get(url)

        # Wait for page to load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(3)

        print("Page loaded, looking for dropdown...")

        # Find the inscription dropdown (button with id=dropdown-1)
        dropdown = find_inscription_dropdown(driver)

        if not dropdown:
            print("❌ No inscription dropdown found!")
            return

        print("✅ Found inscription dropdown, analyzing options...")

        success = handle_custom_dropdown(driver, dropdown, folder)

        if not success:
            print("Failed to process dropdown. Taking screenshots as backup...")
            take_fallback_screenshots(driver, folder)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if 'driver' in locals():
            driver.quit()


def find_inscription_dropdown(driver):
    """Find the inscription dropdown button (ignore language dropdown)"""
    try:
        dropdown = driver.find_element(By.ID, "dropdown-1")
        print("✅ Found inscription dropdown with id=dropdown-1")
        return dropdown
    except NoSuchElementException:
        print("❌ Inscription dropdown (id=dropdown-1) not found")
        return None


def handle_custom_dropdown(driver, dropdown, folder):
    """Handle inscription dropdown (Bootstrap custom dropdown)"""

    print("Handling custom inscription dropdown...")

    try:
        # Click to open dropdown
        driver.execute_script("arguments[0].click();", dropdown)
        time.sleep(1)

        # Find all inscription options
        options = driver.find_elements(By.CSS_SELECTOR, ".dropdown-menu .dropdown-item")

        if not options:
            print("❌ No inscription options found inside dropdown")
            return False

        print(f"✅ Found {len(options)} inscription options")

        total_downloaded = 0

        for i, option in enumerate(options):
            try:
                option_text = option.text.strip()
                if not option_text:
                    continue

                # Extract year from option
                year = extract_year(option_text)

                # Skip if year not within 1000–1400 CE
                if year is None or not (1400 <= year <= 1850):
                    print(f"⏭️ Skipping option {i + 1}/{len(options)}: '{option_text}' (year={year})")
                    continue

                print(f"\n🔄 Processing option {i + 1}/{len(options)}: '{option_text}' (year={year})")

                # Click option
                driver.execute_script("arguments[0].click();", option)
                time.sleep(3)

                wait_for_content_load(driver)

                # Download images
                option_folder = os.path.join(folder, sanitize_filename(option_text))
                downloaded = download_content_images(driver, option_folder, option_text)
                total_downloaded += downloaded

                print(f"✅ Downloaded {downloaded} JPG images for '{option_text}'")

                # Reopen dropdown for next option
                if i < len(options) - 1:
                    driver.execute_script("arguments[0].click();", dropdown)
                    time.sleep(1)
                    # Refresh options list because DOM may change
                    options = driver.find_elements(By.CSS_SELECTOR, ".dropdown-menu .dropdown-item")

            except Exception as e:
                print(f"❌ Error processing option '{option.text}': {e}")

        print(f"\n🎉 Total JPG images downloaded: {total_downloaded}")
        return True

    except Exception as e:
        print(f"Error handling inscription dropdown: {e}")
        return False


def wait_for_content_load(driver):
    """Wait for page content to load after selecting an inscription"""
    try:
        WebDriverWait(driver, 10).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
    except TimeoutException:
        pass


def download_content_images(driver, folder, option_name):
    """Download inscription images from the page (force .jpg)"""
    os.makedirs(folder, exist_ok=True)
    downloaded = 0

    print(f"Looking for inscription images...")

    time.sleep(2)

    all_images = driver.find_elements(By.TAG_NAME, "img")
    print(f"Found {len(all_images)} total images on page")

    content_images = []
    for img in all_images:
        src = img.get_attribute("src") or ""
        if not src:
            continue

        # Filter out obvious UI images
        if any(x in src.lower() for x in ["logo", "icon", "arrow", "menu", "nav", "lang"]):
            continue

        content_images.append(img)

    print(f"Filtered to {len(content_images)} potential inscription images")

    for i, img in enumerate(content_images):
        try:
            src = img.get_attribute("src")
            if not src:
                continue

            if src.startswith("data:image"):
                downloaded += save_base64_image(src, folder, f"{sanitize_filename(option_name)}_img_{i+1}")
            else:
                downloaded += save_url_image(src, driver.current_url, folder,
                                             f"{sanitize_filename(option_name)}_img_{i+1}")
        except Exception as e:
            print(f"❌ Error downloading image {i+1}: {e}")

    return downloaded


def save_base64_image(data_url, folder, filename_prefix):
    """Save base64 image always as .jpg"""
    try:
        header, data = data_url.split(',', 1)
        binary_data = base64.b64decode(data)

        filename = f"{filename_prefix}.jpg"
        filepath = os.path.join(folder, filename)

        with open(filepath, 'wb') as f:
            f.write(binary_data)

        print(f"✅ Saved base64 image as JPG: {filename}")
        return 1
    except Exception as e:
        print(f"❌ Error saving base64 image: {e}")
        return 0


def save_url_image(src, base_url, folder, filename_prefix):
    """Save URL image always as .jpg"""
    try:
        if not src.startswith('http'):
            src = urljoin(base_url, src)

        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Referer': base_url
        }

        response = requests.get(src, headers=headers, timeout=15)
        if response.status_code == 200:
            filename = f"{filename_prefix}.jpg"
            filepath = os.path.join(folder, filename)

            with open(filepath, 'wb') as f:
                f.write(response.content)

            print(f"✅ Saved: {filename}")
            return 1
        else:
            print(f"❌ HTTP {response.status_code} for {src}")
            return 0
    except Exception as e:
        print(f"❌ Error saving URL image: {e}")
        return 0


def sanitize_filename(filename):
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return filename.strip()[:50]


def take_fallback_screenshots(driver, folder):
    try:
        screenshots_folder = os.path.join(folder, 'screenshots')
        os.makedirs(screenshots_folder, exist_ok=True)
        driver.save_screenshot(os.path.join(screenshots_folder, 'full_page.png'))
        print("✅ Saved fallback screenshot")
    except Exception as e:
        print(f"❌ Error taking fallback screenshot: {e}")


def extract_year(text):
    """Extract year from inscription text, return int or None"""
    match = re.search(r'(\d{3,4})\s*CE', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


if __name__ == "__main__":
    github_url = "https://mythicsociety.github.io/AksharaBhandara/#/learn/Shasanagalu"
    print("🚀 Starting GitHub Pages inscription scraper...")
    print("=" * 60)
    scrape_github_pages_dropdown_content(github_url)
