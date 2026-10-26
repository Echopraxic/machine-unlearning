"""Live artifact fetchers for Phase 1 `acquire`. Only exercised during the
online window (Phase 1); never touches guest-only (CVE-target) artifacts,
which are built inside the isolated VM instead (see pipeline/vm/).
"""

import os
import urllib.request
from pathlib import Path

from . import config

# The hf_xet fast-transfer backend generates malformed temp filenames on Windows
# (a literal run of '*' characters where a content hash belongs), which fails
# with WinError 123. Force the plain HTTP transfer path instead. Set before
# huggingface_hub is imported to guarantee it's read at backend-selection time.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def fetch_huggingface(repo: str, revision: str, local_path: Path, repo_type: str) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, revision=revision, repo_type=repo_type,
                      local_dir=str(local_path))


def fetch_github_file(repo: str, commit: str, path: str, local_dir: Path) -> Path:
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    local_dir.mkdir(parents=True, exist_ok=True)
    target = local_dir / Path(path).name
    urllib.request.urlretrieve(url, str(target))  # noqa: S310 -- pinned commit, fixed host
    return target


def fetch_artifact(artifact: dict) -> None:
    """Dispatch by source.kind. Raises on failure — caller handles per-artifact."""
    src = artifact["source"]
    kind = src["kind"]
    local_path = config.REPO_ROOT / artifact["local_path"]

    if kind == "huggingface":
        repo_type = "model" if artifact["type"] == "model" else "dataset"
        fetch_huggingface(src["repo"], src["revision"], local_path, repo_type)
    elif kind == "github":
        fetch_github_file(src["repo"], src["commit"], src["path"], local_path)
    else:
        raise ValueError(f"no fetcher for source.kind={kind!r} (artifact {artifact['id']})")
