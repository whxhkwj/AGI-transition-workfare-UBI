# -*- coding: utf-8 -*-
"""Render the graphical abstract from its HTML source.

The graphical abstract is a bright, professional infographic laid out in
``figures/graphical_abstract.html``. It is rendered with headless Chromium through
Playwright so that every label, symbol and number is set exactly (no rasterised
or garbled text). The PNG is exported at device scale factor 2 for a
high-resolution raster (3200 x 1600 px); the PDF is a single vector page.

Requirements beyond the simulation stack:

    pip install playwright
    # then either: playwright install chromium
    # or point Playwright at a system Chromium via CHROMIUM_PATH below.

The environment variable CHROMIUM_PATH overrides the browser executable.

Outputs:
    paper/graphical_abstract.png   (3200 x 1600 px, 2x)
    paper/graphical_abstract.pdf   (single vector page)

All figures quoted in the infographic (min_t C_t = 0.357 > C^min = 0.30, the
critical re-absorption rate omega_c, g_A = 10%) are produced by the engines in
``simulation/`` and cross-checked by ``analysis/check_ga_panel.py``.
"""
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_HTML = _os.path.join(_HERE, "graphical_abstract.html")
_OUT = _os.path.join(_ROOT, "paper")

# Default to a system Chromium when present; override with CHROMIUM_PATH.
_CHROMIUM = _os.environ.get(
    "CHROMIUM_PATH",
    "/usr/local/bin/chromium" if _os.path.exists("/usr/local/bin/chromium") else None,
)

CANVAS_WIDTH = 1600


def main():
    from pathlib import Path
    from playwright.sync_api import sync_playwright

    url = Path(_HTML).resolve().as_uri()
    launch_kwargs = {"args": ["--no-sandbox", "--disable-gpu"]}
    if _CHROMIUM:
        launch_kwargs["executable_path"] = _CHROMIUM

    with sync_playwright() as p:
        browser = p.chromium.launch(**launch_kwargs)
        page = browser.new_page(
            viewport={"width": CANVAS_WIDTH, "height": 900},
            device_scale_factor=2,
        )
        page.goto(url)
        page.wait_for_timeout(300)
        height = page.evaluate("document.body.scrollHeight")
        page.set_viewport_size({"width": CANVAS_WIDTH, "height": height})
        page.wait_for_timeout(200)

        page.screenshot(path=_os.path.join(_OUT, "graphical_abstract.png"))
        # A few extra px of page height guard against sub-pixel rounding onto a
        # second page.
        page.pdf(
            path=_os.path.join(_OUT, "graphical_abstract.pdf"),
            width=f"{CANVAS_WIDTH}px",
            height=f"{height + 3}px",
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
            print_background=True,
        )
        browser.close()

    print("rendered graphical abstract, body height %d px" % height)
    print(" -> paper/graphical_abstract.png, paper/graphical_abstract.pdf")


if __name__ == "__main__":
    main()
