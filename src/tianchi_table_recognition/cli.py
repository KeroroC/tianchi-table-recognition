"""Command-line interface for the Tianchi 532510 table recognition pipeline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tianchi_table_recognition.config import CLIOverrides, load_config
from tianchi_table_recognition.dataset import DatasetSchemaError
from tianchi_table_recognition.gateway import (
    BailianOpenAITransport,
    ComplianceError,
    ModelTransport,
)
from tianchi_table_recognition.pipeline import export_existing_run, run_pipeline
from tianchi_table_recognition.submission import SubmissionGateError


def _add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--questions-file", type=Path, default=None)
    parser.add_argument("--template-file", type=Path, default=None)
    parser.add_argument("--documents-dir", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="ttr",
        description="Tianchi 532510 multimodal table recognition and QA pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run Stage 1 extraction, Stage 2 QA, and export")
    _add_common_dataset_args(run_parser)
    run_parser.add_argument("--bailian-base-url", type=str, default=None)
    run_parser.add_argument("--vision-extract-model", type=str, default=None)
    run_parser.add_argument("--text-reason-model", type=str, default=None)
    run_parser.add_argument("--ids", type=str, default=None, help="Comma-separated question IDs")
    run_parser.add_argument("--run-id", type=str, default=None)
    run_parser.add_argument("--output", type=Path, default=None)
    run_parser.add_argument("--allow-partial", action="store_true")
    run_parser.add_argument("--no-export", action="store_true")

    export_parser = subparsers.add_parser("export", help="Export Submission from an existing Run")
    _add_common_dataset_args(export_parser)
    export_parser.add_argument("--run-id", type=str, required=True)
    export_parser.add_argument("--output", type=Path, default=None)
    export_parser.add_argument("--allow-partial", action="store_true")

    return parser


def _parse_ids(raw_ids: str | None) -> list[str] | None:
    if not raw_ids:
        return None
    parsed = [item.strip() for item in raw_ids.split(",") if item.strip()]
    return parsed or None


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: ModelTransport | None = None,
) -> int:
    """CLI entrypoint returning process exit status."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    overrides = CLIOverrides(
        dataset_dir=args.dataset_dir,
        questions_file=args.questions_file,
        submission_template_file=args.template_file,
        documents_dir=args.documents_dir,
        work_dir=args.work_dir,
        bailian_base_url=getattr(args, "bailian_base_url", None),
        vision_extract_model=getattr(args, "vision_extract_model", None),
        text_reason_model=getattr(args, "text_reason_model", None),
    )
    config = load_config(cli_overrides=overrides)

    try:
        if args.command == "run":
            active_transport = transport or BailianOpenAITransport(
                base_url=config.bailian_base_url,
                api_key=config.api_key,
            )
            summary = run_pipeline(
                config=config,
                transport=active_transport,
                selected_ids=_parse_ids(args.ids),
                run_id=args.run_id,
                export=not args.no_export,
                allow_partial=args.allow_partial,
                output_path=args.output,
            )
            print(
                f"Run {summary.run_id} finished: "
                f"{summary.succeeded_count}/{summary.total_questions} ok, "
                f"{summary.warnings_count} warnings."
            )
            if summary.submission_result is not None:
                print(
                    f"Exported submission ({summary.submission_result.exported_count}/"
                    f"{summary.submission_result.total_template_count}) to "
                    f"{summary.submission_result.output_path}"
                )
            return 0

        if args.command == "export":
            res = export_existing_run(
                config=config,
                run_id=args.run_id,
                allow_partial=args.allow_partial,
                output_path=args.output,
            )
            print(
                f"Exported submission ({res.exported_count}/{res.total_template_count}) "
                f"to {res.output_path}"
            )
            return 0
    except (SubmissionGateError, ComplianceError, DatasetSchemaError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
