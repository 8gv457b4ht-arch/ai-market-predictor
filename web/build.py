"""Inline src/* into one self-contained file: dist/AI_Market_Predictor.html"""
from pathlib import Path
src = Path(__file__).parent / "src"
html = (src / "index.html").read_text()
for marker, f in (("/*CSS*/", "style.css"), ("/*I18N*/", "i18n.js"), ("/*STATUS*/", "status.js"), ("/*CORE*/", "core.js"),
                  ("/*CONNECTORS*/", "connectors.js"), ("/*APP*/", "app.js")):
    body = (src / f).read_text()
    assert "</script" not in body.lower(), f
    html = html.replace(marker, body, 1)
out = Path(__file__).parent / "dist" / "AI_Market_Predictor.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html)
print(out, len(html) // 1024, "KB")
