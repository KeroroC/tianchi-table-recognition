"""Tests for three-layer configuration loading."""

from __future__ import annotations

from pathlib import Path

from tianchi_table_recognition.config import CLIOverrides, load_config
from tianchi_table_recognition.models import ModelRole


def test_load_config_reads_repo_defaults() -> None:
    cfg = load_config(env={}, cli_overrides=None)

    assert cfg.bailian_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert cfg.model_roles[ModelRole.VISION_EXTRACT] == "qwen-vl-max"
    assert cfg.model_roles[ModelRole.VISION_VERIFY] == "qwen-vl-max"
    assert cfg.model_roles[ModelRole.VISION_TRIAGE] == "qwen-vl-plus"
    assert cfg.model_roles[ModelRole.TEXT_REASON] == "qwen-max"
    assert cfg.work_dir == Path("work")
    assert cfg.api_key is None
    assert cfg.pdf_render_dpi == 300
    assert cfg.pdf_triage_page_threshold == 8
    assert cfg.pdf_triage_thumbnail_dpi == 72


def test_env_overrides_repo_defaults(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "ext_dataset"
    work_dir = tmp_path / "local_work"
    env = {
        "TTR_DATASET_DIR": str(dataset_dir),
        "TTR_QUESTIONS_FILE": "custom_questions.xlsx",
        "TTR_TEMPLATE_FILE": "custom_template.xlsx",
        "TTR_DOCUMENTS_DIR": "docs",
        "TTR_WORK_DIR": str(work_dir),
        "TTR_BAILIAN_BASE_URL": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "TTR_MODEL_VISION_EXTRACT": "qwen2.5-vl-72b-instruct",
        "TTR_MODEL_TEXT_REASON": "qwen-plus",
        "DASHSCOPE_API_KEY": "sk-test-env-key",
        "TTR_PDF_RENDER_DPI": "240",
        "TTR_PDF_TRIAGE_PAGE_THRESHOLD": "12",
        "TTR_PDF_TRIAGE_THUMBNAIL_DPI": "96",
    }

    cfg = load_config(env=env)

    assert cfg.dataset_dir == dataset_dir
    assert cfg.questions_path == dataset_dir / "custom_questions.xlsx"
    assert cfg.submission_template_path == dataset_dir / "custom_template.xlsx"
    assert cfg.documents_dir_path == dataset_dir / "docs"
    assert cfg.work_dir == work_dir
    assert cfg.bailian_base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert cfg.model_roles[ModelRole.VISION_EXTRACT] == "qwen2.5-vl-72b-instruct"
    assert cfg.model_roles[ModelRole.TEXT_REASON] == "qwen-plus"
    assert cfg.api_key == "sk-test-env-key"
    assert cfg.pdf_render_dpi == 240
    assert cfg.pdf_triage_page_threshold == 12
    assert cfg.pdf_triage_thumbnail_dpi == 96


def test_cli_overrides_env_without_clobbering_unset_fields(tmp_path: Path) -> None:
    env_dataset = tmp_path / "env_data"
    cli_work = tmp_path / "cli_work"
    abs_questions = tmp_path / "abs_questions.xlsx"
    env = {
        "TTR_DATASET_DIR": str(env_dataset),
        "TTR_WORK_DIR": str(tmp_path / "env_work"),
        "TTR_MODEL_VISION_EXTRACT": "qwen-vl-plus",
        "TTR_PDF_RENDER_DPI": "240",
        "TTR_PDF_TRIAGE_PAGE_THRESHOLD": "10",
    }
    overrides = CLIOverrides(
        work_dir=cli_work,
        questions_file=abs_questions,
        vision_extract_model=None,
        pdf_render_dpi=300,
        pdf_triage_page_threshold=5,
    )

    cfg = load_config(env=env, cli_overrides=overrides)

    assert cfg.dataset_dir == env_dataset
    assert cfg.work_dir == cli_work
    assert cfg.questions_path == abs_questions
    assert cfg.model_roles[ModelRole.VISION_EXTRACT] == "qwen-vl-plus"
    assert cfg.pdf_render_dpi == 300
    assert cfg.pdf_triage_page_threshold == 5
