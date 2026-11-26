"""Three-layer configuration loader (repo defaults -> env vars -> CLI overrides)."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tianchi_table_recognition.models import ModelRole

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "default.toml"


class PricingRate(BaseModel):
    """Token pricing rates in CNY per 1,000 tokens."""

    model_config = ConfigDict(frozen=True)

    input: float = Field(ge=0.0)
    output: float = Field(ge=0.0)


class CLIOverrides(BaseModel):
    """Optional CLI-level overrides; None values preserve lower-precedence layers."""

    model_config = ConfigDict(frozen=True)

    dataset_dir: Path | None = None
    questions_file: Path | None = None
    submission_template_file: Path | None = None
    documents_dir: Path | None = None
    work_dir: Path | None = None
    bailian_base_url: str | None = None
    vision_extract_model: str | None = None
    text_reason_model: str | None = None
    pdf_render_dpi: int | None = None
    pdf_triage_page_threshold: int | None = None


class AppConfig(BaseModel):
    """Resolved runtime configuration for the pipeline."""

    model_config = ConfigDict(frozen=True)

    bailian_base_url: str
    work_dir: Path
    dataset_dir: Path | None
    questions_file: Path
    submission_template_file: Path
    documents_dir: Path
    model_roles: dict[ModelRole, str]
    pricing_cny_per_1k_tokens: dict[str, PricingRate]
    api_key: str | None = None
    pdf_render_dpi: int = Field(default=300, ge=72)
    pdf_triage_page_threshold: int = Field(default=8, ge=1)
    pdf_triage_thumbnail_dpi: int = Field(default=72, ge=36)

    def resolve_dataset_path(self, raw_path: Path) -> Path:
        """Resolve a relative dataset path against dataset_dir if configured."""
        if raw_path.is_absolute() or self.dataset_dir is None:
            return raw_path
        return self.dataset_dir / raw_path

    @property
    def questions_path(self) -> Path:
        """Resolved path to the competition questions spreadsheet."""
        return self.resolve_dataset_path(self.questions_file)

    @property
    def submission_template_path(self) -> Path:
        """Resolved path to the competition submission template spreadsheet."""
        return self.resolve_dataset_path(self.submission_template_file)

    @property
    def documents_dir_path(self) -> Path:
        """Resolved directory containing Source Documents."""
        return self.resolve_dataset_path(self.documents_dir)

    def model_for_role(self, role: ModelRole) -> str:
        """Return configured model ID for a logical Model Role."""
        return self.model_roles[role]

    def pricing_for_model(self, model_id: str) -> PricingRate:
        """Return token pricing for model_id, falling back to 'default'."""
        return self.pricing_cny_per_1k_tokens.get(
            model_id,
            self.pricing_cny_per_1k_tokens.get("default", PricingRate(input=0.0, output=0.0)),
        )


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _apply_env(
    values: dict[str, Any],
    model_roles: dict[ModelRole, str],
    env_map: Mapping[str, str],
) -> None:
    str_keys = {
        "TTR_BAILIAN_BASE_URL": "bailian_base_url",
        "TTR_WORK_DIR": "work_dir",
        "TTR_DATASET_DIR": "dataset_dir",
        "TTR_QUESTIONS_FILE": "questions_file",
        "TTR_TEMPLATE_FILE": "submission_template_file",
        "TTR_DOCUMENTS_DIR": "documents_dir",
        "DASHSCOPE_API_KEY": "api_key",
    }
    for env_key, target_key in str_keys.items():
        if raw := env_map.get(env_key):
            values[target_key] = raw

    role_env_keys = {
        ModelRole.VISION_EXTRACT: "TTR_MODEL_VISION_EXTRACT",
        ModelRole.VISION_VERIFY: "TTR_MODEL_VISION_VERIFY",
        ModelRole.VISION_TRIAGE: "TTR_MODEL_VISION_TRIAGE",
        ModelRole.TEXT_REASON: "TTR_MODEL_TEXT_REASON",
    }
    for role, env_key in role_env_keys.items():
        if raw := env_map.get(env_key):
            model_roles[role] = raw

    int_keys = {
        "TTR_PDF_RENDER_DPI": "pdf_render_dpi",
        "TTR_PDF_TRIAGE_PAGE_THRESHOLD": "pdf_triage_page_threshold",
        "TTR_PDF_TRIAGE_THUMBNAIL_DPI": "pdf_triage_thumbnail_dpi",
    }
    for env_key, target_key in int_keys.items():
        if raw := env_map.get(env_key):
            values[target_key] = int(raw)


def _apply_cli(
    values: dict[str, Any],
    model_roles: dict[ModelRole, str],
    cli: CLIOverrides,
) -> None:
    cli_map: list[tuple[str, Any]] = [
        ("bailian_base_url", cli.bailian_base_url),
        ("work_dir", cli.work_dir),
        ("dataset_dir", cli.dataset_dir),
        ("questions_file", cli.questions_file),
        ("submission_template_file", cli.submission_template_file),
        ("documents_dir", cli.documents_dir),
        ("pdf_render_dpi", cli.pdf_render_dpi),
        ("pdf_triage_page_threshold", cli.pdf_triage_page_threshold),
    ]
    for target_key, val in cli_map:
        if val is not None:
            values[target_key] = val

    role_cli_map = {
        ModelRole.VISION_EXTRACT: cli.vision_extract_model,
        ModelRole.TEXT_REASON: cli.text_reason_model,
    }
    for role, val in role_cli_map.items():
        if val is not None:
            model_roles[role] = val


def load_config(
    config_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    cli_overrides: CLIOverrides | None = None,
) -> AppConfig:
    """Load configuration merging repo defaults -> environment variables -> CLI overrides."""
    raw = _load_toml(config_path or _DEFAULT_CONFIG_PATH)

    values: dict[str, Any] = {
        "bailian_base_url": raw["bailian_base_url"],
        "work_dir": raw["work_dir"],
        "dataset_dir": raw.get("dataset_dir") or None,
        "questions_file": raw["questions_file"],
        "submission_template_file": raw["submission_template_file"],
        "documents_dir": raw["documents_dir"],
        "pdf_render_dpi": raw.get("pdf_render_dpi", 300),
        "pdf_triage_page_threshold": raw.get("pdf_triage_page_threshold", 8),
        "pdf_triage_thumbnail_dpi": raw.get("pdf_triage_thumbnail_dpi", 72),
        "api_key": None,
    }

    raw_roles = raw.get("model_roles", {})
    model_roles: dict[ModelRole, str] = {
        role: str(raw_roles[role.value]) for role in ModelRole
    }

    env_map = os.environ if env is None else env
    _apply_env(values, model_roles, env_map)

    if cli_overrides is not None:
        _apply_cli(values, model_roles, cli_overrides)

    raw_pricing = raw.get("pricing_cny_per_1k_tokens", {})
    pricing = {
        name: PricingRate.model_validate(rate) for name, rate in raw_pricing.items()
    }

    dataset_dir_raw = values["dataset_dir"]
    return AppConfig(
        bailian_base_url=str(values["bailian_base_url"]),
        work_dir=Path(values["work_dir"]),
        dataset_dir=Path(dataset_dir_raw) if dataset_dir_raw else None,
        questions_file=Path(values["questions_file"]),
        submission_template_file=Path(values["submission_template_file"]),
        documents_dir=Path(values["documents_dir"]),
        model_roles=model_roles,
        pricing_cny_per_1k_tokens=pricing,
        api_key=str(values["api_key"]) if values["api_key"] else None,
        pdf_render_dpi=int(values["pdf_render_dpi"]),
        pdf_triage_page_threshold=int(values["pdf_triage_page_threshold"]),
        pdf_triage_thumbnail_dpi=int(values["pdf_triage_thumbnail_dpi"]),
    )
