"""Load config/settings.yaml, the single source of every tunable parameter."""

from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_SETTINGS_PATH = PACKAGE_ROOT / "config" / "settings.yaml"


def load_settings(settings_path: str | Path | None = None) -> dict:
    """Read the YAML settings file and return it as a nested dict."""
    path = Path(settings_path) if settings_path else DEFAULT_SETTINGS_PATH
    with path.open() as settings_file:
        return yaml.safe_load(settings_file)


def resolve_project_path(path_setting: str | Path) -> Path:
    """Turn a path from settings.yaml into an absolute path (relative paths are anchored at the project root)."""
    path = Path(path_setting).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
