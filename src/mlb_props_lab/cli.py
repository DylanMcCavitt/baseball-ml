from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mlb_props_lab.dashboard import build_dashboard
from mlb_props_lab.feature_registry import load_registry, validate_registry
from mlb_props_lab.reports import generate_feature_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlb-props-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    registry_parser = subparsers.add_parser("feature-registry")
    registry_subparsers = registry_parser.add_subparsers(dest="registry_command", required=True)
    validate_parser = registry_subparsers.add_parser("validate")
    validate_parser.add_argument("--path", type=Path, default=None)

    report_parser = subparsers.add_parser("report")
    report_subparsers = report_parser.add_subparsers(dest="report_command", required=True)
    features_parser = report_subparsers.add_parser("features")
    features_parser.add_argument("--issue", required=True)
    features_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports"))
    features_parser.add_argument("--run-id", default=None)
    features_parser.add_argument("--registry-path", type=Path, default=None)

    dashboard_parser = subparsers.add_parser("dashboard")
    dashboard_parser.add_argument("--reports-root", type=Path, default=Path("artifacts/reports"))
    dashboard_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/dashboard/index.html"),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "feature-registry":
        return _handle_feature_registry(args)
    if args.command == "report":
        return _handle_report(args)
    if args.command == "dashboard":
        path = build_dashboard(args.reports_root, args.output)
        print(f"Dashboard written to {path}")
        return 0
    return 2


def _handle_feature_registry(args: argparse.Namespace) -> int:
    data = load_registry(args.path)
    result = validate_registry(data)
    if not result.ok:
        for error in result.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "Feature registry valid: "
        f"{result.feature_count} features, "
        f"{result.family_count} families, "
        f"{result.required_feature_count} v1 required"
    )
    return 0


def _handle_report(args: argparse.Namespace) -> int:
    if args.report_command == "features":
        path = generate_feature_report(
            issue=args.issue,
            output_dir=args.output_dir,
            run_id=args.run_id,
            registry_path=args.registry_path,
        )
        print(f"Feature report written to {path}")
        return 0
    return 2
