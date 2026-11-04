"""Optional browser checks; set QEV_TEST_BROWSER=1 after installing Playwright's Chromium."""
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest

pytestmark = pytest.mark.browser
INJECTION = '<img src=x onerror="window.qevInjected=true">'


@pytest.fixture(scope="module")
def browser():
    if os.environ.get("QEV_TEST_BROWSER") != "1":
        pytest.skip("set QEV_TEST_BROWSER=1 to run browser checks")
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("QEV_TEST_BROWSER_EXECUTABLE"))
        yield browser
        browser.close()


def open_page(browser, data, status=200):
    page = browser.new_page()
    html = (Path(__file__).parents[1] / "qev/playground/index.html").read_text(encoding="utf-8")

    def respond(route):
        path = urlsplit(route.request.url).path
        if path == "/":
            route.fulfill(body=html, content_type="text/html")
        elif path == "/v1/systemone":
            route.fulfill(json=data, status=status)
        else:
            route.abort()

    page.route("http://qev.test/**", respond)
    page.goto("http://qev.test/")
    return page


def test_question_names_choices_legends_and_metadata_are_text(browser):
    data = {
        "model": INJECTION,
        "answers": {
            INJECTION: {"type": "noul", "noul": 0.7},
            "choice": {"type": "choice", "choice": INJECTION, "probabilities": {INJECTION: 0.6, "other": 0.4}, "confidence": 0.2},
            "score": {"type": "score", "score": 0.6, "legend": {"0": INJECTION, "1": "fine"},
                      "probabilities": {"0": 0.4, "1": 0.6}, "confidence": 0.2},
        },
        "usage": {"input_tokens": 10}, "qev": {"branches": 3, "fork_mode": INJECTION},
    }
    page = open_page(browser, data)
    try:
        page.click("#run")
        page.wait_for_selector("#out .ans")
        assert page.locator("#out img, #out svg, #out script").count() == 0
        assert INJECTION in page.locator("#out").inner_text()
        assert page.evaluate("window.qevInjected === undefined")
        assert page.locator("#out .ans").count() == 3
    finally:
        page.close()


def test_api_errors_are_rendered_as_text(browser):
    page = open_page(browser, {"detail": INJECTION}, status=422)
    try:
        page.click("#run")
        page.wait_for_selector("#out .err")
        assert page.locator("#out img").count() == 0
        assert json.loads(page.locator("#out").inner_text().partition(": ")[2])["detail"] == INJECTION
    finally:
        page.close()


def test_invalid_question_json_does_not_execute_html(browser):
    page = open_page(browser, {})
    try:
        page.fill("#questions", INJECTION)
        page.click("#run")
        page.wait_for_selector("#out .err")
        assert page.locator("#out img").count() == 0
        assert "not valid JSON" in page.locator("#out").inner_text()
    finally:
        page.close()
