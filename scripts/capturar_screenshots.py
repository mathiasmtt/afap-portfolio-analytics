"""Script one-shot para capturar screenshots del dashboard Streamlit.

Asume que el server está corriendo en http://127.0.0.1:8501.
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
URL = "http://127.0.0.1:8501"

VIEWPORT = {"width": 1440, "height": 900}


def capturar() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport=VIEWPORT,
            device_scale_factor=2,  # retina quality
        )
        page = context.new_page()
        page.goto(URL, wait_until="networkidle")

        # Esperar a que renderice los gráficos Plotly.
        page.wait_for_timeout(4000)

        # Screenshot 1: scroll al capítulo 1 para asegurar renderizado.
        page.evaluate(
            """
            () => {
                const nodes = Array.from(document.querySelectorAll('div'));
                const target = nodes.find(n => (n.textContent || '').trim()
                    .startsWith('Capítulo 1 · Describir'));
                if (target) target.scrollIntoView({block: 'start'});
            }
            """
        )
        page.wait_for_timeout(2500)
        page.screenshot(
            path=str(OUT_DIR / "01_overview.png"),
            full_page=False,
        )
        print(f"✓ Guardado: {OUT_DIR / '01_overview.png'}")

        # Screenshot 2: scatter del capítulo 5 (la lista dorada).
        # Hago scroll hasta el scatter. Busco el texto "Capítulo 5 · Cuantificar".
        page.evaluate(
            """
            () => {
                const nodes = Array.from(document.querySelectorAll('div'));
                const target = nodes.find(n => (n.textContent || '').trim()
                    .startsWith('Capítulo 5 · Cuantificar'));
                if (target) target.scrollIntoView({block: 'start'});
            }
            """
        )
        page.wait_for_timeout(2500)  # esperar redraw
        page.screenshot(
            path=str(OUT_DIR / "02_lista_dorada.png"),
            full_page=False,
        )
        print(f"✓ Guardado: {OUT_DIR / '02_lista_dorada.png'}")

        browser.close()


if __name__ == "__main__":
    capturar()
