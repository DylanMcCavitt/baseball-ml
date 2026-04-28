from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from typing import Any


def discover_manifests(reports_root: str | Path) -> list[Path]:
    root = Path(reports_root)
    if not root.exists():
        return []
    return sorted(root.glob("*/*/manifest.json"))


def build_dashboard(
    reports_root: str | Path = "artifacts/reports",
    output: str | Path = "artifacts/dashboard/index.html",
) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifests = [_load_manifest(path) for path in discover_manifests(reports_root)]
    output_path.write_text(_render_dashboard(manifests, output_path.parent), encoding="utf-8")
    return output_path


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest["_manifest_path"] = path
    return manifest


def _render_dashboard(manifests: list[dict[str, Any]], output_parent: Path) -> str:
    cards = []
    for manifest in manifests:
        manifest_path = Path(manifest["_manifest_path"])
        report_path = manifest_path.parent / manifest.get("report", "report.md")
        family_rows = []
        for family, counts in manifest.get("families", {}).items():
            family_rows.append(
                "<tr>"
                f"<td>{escape(family)}</td>"
                f"<td>{counts.get('v1_required', 0)}</td>"
                f"<td>{counts.get('v1_optional', 0)}</td>"
                f"<td>{counts.get('later', 0)}</td>"
                f"<td>{counts.get('total', 0)}</td>"
                "</tr>"
            )
        visual_links = []
        for visual in manifest.get("visuals", []):
            visual_path = manifest_path.parent / visual
            visual_links.append(
                f'<a href="{escape(_relative_link(visual_path, output_parent))}">'
                f"{escape(Path(visual).name)}</a>"
            )
        cards.append(
            f"""
            <section class="card">
              <h2>{escape(manifest.get("issue", "unknown issue"))}</h2>
              <p><strong>Run:</strong> {escape(manifest.get("run_id", "unknown"))}</p>
              <p><strong>Type:</strong> {escape(manifest.get("report_type", "unknown"))}</p>
              <p><strong>Features:</strong> {manifest.get("feature_count", 0)}
                 total, {manifest.get("required_feature_count", 0)} v1 required</p>
              <p><a href="{escape(_relative_link(report_path, output_parent))}">Open report</a></p>
              <div class="visuals">{' '.join(visual_links)}</div>
              <table>
                <thead>
                  <tr><th>Family</th><th>Required</th><th>Optional</th><th>Later</th><th>Total</th></tr>
                </thead>
                <tbody>{''.join(family_rows)}</tbody>
              </table>
            </section>
            """
        )
    empty = "<p>No report manifests found yet.</p>" if not cards else ""
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>MLB Props Reboot Dashboard</title>
    <style>
      body {{
        margin: 0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #f6f8fa;
        color: #17202a;
      }}
      header {{
        padding: 28px 36px;
        background: #ffffff;
        border-bottom: 1px solid #d7dee5;
      }}
      main {{
        padding: 24px 36px 48px;
        display: grid;
        gap: 18px;
      }}
      .card {{
        background: #ffffff;
        border: 1px solid #d7dee5;
        border-radius: 8px;
        padding: 18px;
      }}
      h1, h2 {{ margin: 0 0 12px; }}
      table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
      th, td {{ text-align: left; border-bottom: 1px solid #edf1f4; padding: 8px; }}
      .visuals a {{ display: inline-block; margin: 0 10px 8px 0; }}
    </style>
  </head>
  <body>
    <header>
      <h1>MLB Props Reboot Dashboard</h1>
      <p>
        Report-backed review surface for data, feature, model, market,
        and paper candidate work.
      </p>
    </header>
    <main>
      {empty}
      {''.join(cards)}
    </main>
  </body>
</html>
"""


def _relative_link(target: Path, base: Path) -> str:
    return os.path.relpath(target, start=base).replace(os.sep, "/")
