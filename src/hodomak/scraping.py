"""
Scraping module for Hodomak.

Scrapes products from:
- No Clue:   https://shop-noclue.com/collections/all-products
- Juvenile:  https://juvenileeg.com/collections/all

Saves combined results to a CSV (default: data/raw/hedomak_products.csv)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Dict

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]        # project root (hodomak/)
RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CSV_PATH = RAW_DIR / "hedomak_products.csv"


# -------------------------------------------------------------------
# WebDriver factory
# -------------------------------------------------------------------

def make_driver(headless: bool = True) -> webdriver.Chrome:
    """Create and return a configured Chrome WebDriver."""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=chrome_options)
    return driver


# -------------------------------------------------------------------
# No Clue scraper
# -------------------------------------------------------------------

def scrape_no_clue(driver: webdriver.Chrome) -> List[Dict]:
    """
    Scrape products from No Clue.

    URL: https://shop-noclue.com/collections/all-products
    """
    url = "https://shop-noclue.com/collections/all-products"
    driver.get(url)

    # Wait for JavaScript to load
    time.sleep(5)

    products: List[Dict] = []
    visited_cards = set()

    # Find product cards (ensure they are present)
    while True:
        try:
            product_cards = driver.find_elements(By.CSS_SELECTOR, "div.card__content")
            break
        except Exception:
            time.sleep(2)
            continue

    for i in range(len(product_cards)):
        try:
            # Refresh elements (DOM can change after navigation)
            product_cards = driver.find_elements(By.CSS_SELECTOR, "div.card__content")
            product_link = product_cards[i].find_element(By.CSS_SELECTOR, "a.full-unstyled-link")
            product_url = product_link.get_attribute("href")
            print("[No Clue]", product_url)

            # Click product and wait for new page to load
            driver.execute_script("arguments[0].click();", product_link)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h1"))
            )

            # Extract product details
            title = driver.find_element(
                By.CSS_SELECTOR, "div.product__title h1"
            ).text.strip()
            if title in visited_cards:
                driver.back()
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.card__content"))
                )
                continue
            visited_cards.add(title)
            print("[No Clue] Title:", title)

            brand = driver.find_element(
                By.CSS_SELECTOR, "img.header__heading-logo"
            ).get_attribute("alt")
            print("[No Clue] Brand:", brand)

            # Extract regular price (if available)
            regular_price_element = driver.find_elements(
                By.CSS_SELECTOR,
                "div.price__regular span.price-item.price-item--regular",
            )
            regular_price = regular_price_element[0].text.strip() if regular_price_element else ""

            # Extract sale price (if available)
            sale_price_element = driver.find_elements(
                By.CSS_SELECTOR,
                "div.price__sale span.price-item.price-item--sale.price-item--last",
            )
            sale_price = sale_price_element[0].text.strip() if sale_price_element else ""

            # If a sale price exists but no regular price, the regular price
            # might be in the crossed-out <s> tag
            if sale_price and not regular_price:
                crossed_out_price_element = driver.find_elements(
                    By.CSS_SELECTOR,
                    "div.price__sale s.price-item.price-item--regular",
                )
                regular_price = (
                    crossed_out_price_element[0].text.strip()
                    if crossed_out_price_element
                    else ""
                )

            print("[No Clue] Regular:", regular_price)
            print("[No Clue] Sale:   ", sale_price)

            # Description
            description = driver.find_element(
                By.CSS_SELECTOR, "div.product__description p"
            ).text.strip()
            print("[No Clue] Description:", description[:80], "...")

            # Sizes
            sizes_list = []
            size_elements = driver.find_elements(
                By.CSS_SELECTOR, "fieldset input[name='Size']"
            )
            for size_element in size_elements:
                size_value = size_element.get_attribute("value")
                size_class = size_element.get_attribute("class") or ""
                if "disabled" not in size_class:  # Only add if it's available
                    sizes_list.append(size_value)

            sizes = ",".join(sizes_list)
            print("[No Clue] Sizes:", sizes)

            available = 0 if sizes == "" else 1

            # Image
            image_url = driver.find_element(
                By.CSS_SELECTOR, "div.product__media img"
            ).get_attribute("src")
            print("[No Clue] Image:", image_url)

            products.append(
                {
                    "brand": brand,
                    "title": title,
                    "product_url": product_url,
                    "regular_price": regular_price,
                    "sale_price": sale_price,
                    "description": description,
                    "sizes": sizes,
                    "available": available,
                    "image_url": image_url,
                    "source": "no_clue",
                }
            )

            # Go back and wait for products page to reload
            driver.back()
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.card__content"))
            )

        except Exception as e:
            print(f"[No Clue] Error scraping a product: {e}")
            driver.back()
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.card__content"))
            )
            continue

    return products


# -------------------------------------------------------------------
# Juvenile scraper
# -------------------------------------------------------------------

def scrape_juvenile(driver: webdriver.Chrome) -> List[Dict]:
    """
    Scrape products from Juvenile.

    URL pattern: https://juvenileeg.com/collections/all?page={page_number}
    """
    page_number = 1
    products: List[Dict] = []
    visited_cards = set()

    while True:
        # Open the webpage
        url = f"https://juvenileeg.com/collections/all?page={page_number}"
        print(f"[Juvenile] Visiting: {url}")
        driver.get(url)

        # Wait for JavaScript to load
        time.sleep(10)

        # Find product cards (ensure they are present)
        while True:
            try:
                product_cards = driver.find_elements(By.CSS_SELECTOR, "div.block-inner-inner")
                break
            except Exception:
                time.sleep(2)
                continue

        print("[Juvenile] Found cards:", len(product_cards))

        if not product_cards:
            print("[Juvenile] No product cards found, stopping.")
            break

        for i in range(len(product_cards)):
            print("[Juvenile] Scraping product", i)
            try:
                product_cards = driver.find_elements(
                    By.CSS_SELECTOR, "div.block-inner-inner"
                )  # Refresh elements
                product_link = product_cards[i].find_element(
                    By.CSS_SELECTOR, "a.product-link"
                )
                product_url = product_link.get_attribute("href")
                print("[Juvenile] URL:", product_url)

                # Click product and wait for new page to load
                driver.execute_script("arguments[0].click();", product_link)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "h1"))
                )

                # Extract product details
                time.sleep(5)
                title = driver.find_element(
                    By.CSS_SELECTOR, "h1.title"
                ).text.strip()
                if title in visited_cards:
                    print("[Juvenile] Already exists:", title)
                    driver.back()
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "div.block-inner-inner")
                        )
                    )
                    continue
                visited_cards.add(title)

                if title == "":
                    print("[Juvenile] Empty title, aborting.")
                    # you had driver.quit() in notebook; here we just skip
                    driver.back()
                    continue

                print("[Juvenile] Title:", title)

                brand = driver.find_element(
                    By.CSS_SELECTOR, "img.logo__image"
                ).get_attribute("alt")
                print("[Juvenile] Brand:", brand)

                # Prices
                try:
                    sale_price = driver.find_element(
                        By.CSS_SELECTOR, "span.current-price.theme-money"
                    ).text.strip()
                except Exception:
                    sale_price = ""

                try:
                    regular_price = driver.find_element(
                        By.CSS_SELECTOR, "span.was-price.theme-money"
                    ).text.strip()
                except Exception:
                    regular_price = sale_price  # If no sale, set regular price to sale price

                # Ensure both prices are the same if no discount is applied
                if not regular_price:
                    regular_price = sale_price

                print("[Juvenile] Regular:", regular_price)
                print("[Juvenile] Sale:   ", sale_price)

                # Description
                try:
                    description = driver.find_element(
                        By.CSS_SELECTOR, "div.product-description.rte.cf span"
                    ).text.strip()
                except Exception:
                    try:
                        description = driver.find_element(
                            By.CSS_SELECTOR, "div.product-description.rte.cf p"
                        ).text.strip()
                    except Exception:
                        description = ""
                print("[Juvenile] Description:", description[:80], "...")

                # Sizes
                size_buttons = driver.find_elements(
                    By.CSS_SELECTOR, "div.option-selector__btns input.opt-btn"
                )
                sizes_list = []
                for button in size_buttons:
                    # Check if the size is available (does not have 'is-unavailable' class)
                    if "is-unavailable" not in (button.get_attribute("class") or ""):
                        # Find the corresponding label text
                        label = button.find_element(
                            By.XPATH,
                            "./following-sibling::label/span[@class='opt-label__text']",
                        )
                        sizes_list.append(label.text.strip())

                sizes = ",".join(sizes_list)
                print("[Juvenile] Sizes:", sizes)

                available = 0 if sizes == "" else 1
                print(f"[Juvenile] Available: {'Yes' if available == 1 else 'No'}")

                # Image URL
                image_element = driver.find_element(
                    By.CSS_SELECTOR, "a.show-gallery"
                )
                image_url = image_element.get_attribute("href")
                print("[Juvenile] Image:", image_url)
                print()

                products.append(
                    {
                        "brand": brand,
                        "title": title,
                        "product_url": product_url,
                        "regular_price": regular_price,
                        "sale_price": sale_price,
                        "description": description,
                        "sizes": sizes,
                        "available": available,
                        "image_url": image_url,
                        "source": "juvenile",
                    }
                )

                # Go back and wait for products page to reload
                driver.back()
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "div.block-inner-inner")
                    )
                )

            except Exception as e:
                print(f"[Juvenile] Error scraping a product: {e}")
                driver.back()
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "div.block-inner-inner")
                    )
                )
                continue

        # Try to find the "Next" button; stop if not found
        try:
            driver.find_element(By.CSS_SELECTOR, "a.pagination__next")
            page_number += 1
        except Exception:
            print("[Juvenile] No more pages. Stopping.")
            break

    return products


# -------------------------------------------------------------------
# Orchestrator
# -------------------------------------------------------------------

def scrape_all(
    output_csv: str | Path | None = None,
    headless: bool = True,
    include_no_clue: bool = True,
    include_juvenile: bool = True,
) -> Path:
    """
    Run all scrapers and save results to a single CSV.

    Parameters
    ----------
    output_csv : str or Path, optional
        Output CSV path; if None, defaults to data/raw/hedomak_products.csv
    headless : bool
        Run Chrome in headless mode.
    include_no_clue : bool
        Whether to scrape No Clue.
    include_juvenile : bool
        Whether to scrape Juvenile.

    Returns
    -------
    Path
        Path to the output CSV.
    """
    if output_csv is None:
        output_csv = DEFAULT_CSV_PATH
    else:
        output_csv = Path(output_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    driver = make_driver(headless=headless)
    products: List[Dict] = []

    try:
        if include_no_clue:
            products.extend(scrape_no_clue(driver))
        if include_juvenile:
            products.extend(scrape_juvenile(driver))
    finally:
        driver.quit()

    if not products:
        print("No products scraped. Nothing to write.")
        return output_csv

    df = pd.DataFrame(products)
    csv_file = str(output_csv)

    # Append if file exists; otherwise write with header
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_file, mode="w", header=True, index=False)

    print(f"Scraping complete. Saved {len(df)} rows to {csv_file}")
    return output_csv


if __name__ == "__main__":
    scrape_all()
