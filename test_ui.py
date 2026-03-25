from playwright.sync_api import Page, expect, sync_playwright
import os

def verify_feature(page: Page):
  page.goto(f"file://{os.getcwd()}/tools/splatlens_viewer.html")
  page.wait_for_timeout(500)

  # Check Play/Pause
  expect(page.locator("#btn-play")).to_have_attribute("aria-pressed", "false")
  expect(page.locator("#btn-pause")).to_have_attribute("aria-pressed", "true")

  # Check Layers
  expect(page.locator("#btn-field")).to_have_attribute("aria-pressed", "true")
  expect(page.locator("#btn-trail")).to_have_attribute("aria-pressed", "true")

  # Click to change layer and verify aria-pressed updates to false
  page.locator("#btn-field").click()
  page.wait_for_timeout(500)
  expect(page.locator("#btn-field")).to_have_attribute("aria-pressed", "false")
  page.locator("#btn-field").click() # Restore

  # Check Settings Panel Theme buttons
  page.locator("#settings-btn").click()
  page.wait_for_timeout(500)
  expect(page.locator("#btn-dark")).to_have_attribute("aria-pressed", "true")
  expect(page.locator("#btn-light")).to_have_attribute("aria-pressed", "false")

  # Take a screenshot
  page.screenshot(path="/home/jules/verification/verification.png")
  page.wait_for_timeout(1000)

if __name__ == "__main__":
  os.makedirs("/home/jules/verification/video", exist_ok=True)
  with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(record_video_dir="/home/jules/verification/video")
    page = context.new_page()
    try:
      verify_feature(page)
    finally:
      context.close()
      browser.close()
