"""Workspace path and dataset discovery utilities."""

from pathlib import Path
from typing import Iterable, List, Optional

EXCLUDED_DIR_NAMES = {".venv", "venv", ".git", "__pycache__", ".ipynb_checkpoints", "reports"}


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """Walk upward from start_path (or cwd) to locate the project root directory."""
    current = (start_path or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "dataset").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate project root containing 'dataset/' and 'src/' starting from {current}."
    )


def discover_dataset_csvs(
    project_root: Path,
    excluded_dirs: Iterable[str] = EXCLUDED_DIR_NAMES,
) -> List[Path]:
    """Discover CSV files inside the project workspace without assuming filenames."""
    excluded = set(excluded_dirs)
    discovered: List[Path] = []
    for path in sorted(project_root.rglob("*.csv")):
        rel_parts = path.relative_to(project_root).parts
        if any(part in excluded for part in rel_parts):
            continue
        discovered.append(path)
    # Prioritize CSV files inside dataset/ before any other workspace directory.
    discovered.sort(key=lambda p: (0 if p.relative_to(project_root).parts[0] == "dataset" else 1, str(p)))
    return discovered


def get_primary_dataset_path(project_root: Optional[Path] = None) -> Path:
    """Return the first discovered dataset CSV inside the workspace."""
    root = project_root or find_project_root()
    csv_files = discover_dataset_csvs(root)
    if not csv_files:
        raise FileNotFoundError(f"No dataset CSV files found under {root}.")
    return csv_files[0]
