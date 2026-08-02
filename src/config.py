"""
Single source of configuration: loads config.yaml, layers .env secrets on top, and
resolves every on-disk path from docs/DATA_SCHEMA.md section 5 in one place.

In:  config.yaml (relocate with $SCIAGENT_CONFIG) and a .env holding OPENROUTER_API_KEY.
Out: module-level `CFG` with attribute access (CFG.retrieval.k_retrieve), `CFG.paths`
     of absolute Paths, and prompt loading. Unknown keys raise instead of returning None.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import yaml
from dotenv import load_dotenv

REPO_ROOT: Path = Path(__file__).resolve().parents[1]

CONFIG_PATH_ENV_VAR = "SCIAGENT_CONFIG"
API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
APP_NAME_ENV_VAR = "OPENROUTER_APP_NAME"

REQUIRED_SECTIONS = (
    "llm",
    "collection",
    "extraction",
    "chunking",
    "embedding",
    "retrieval",
    "nli",
    "analysis",
    "graph",
    "agent",
    "paths",
    "eval",
    # `compute` is listed here for the same reason as every other section: a config
    # without it used to pass validation and then fail much later inside
    # resolve_device() with an AttributeError that never named the offending file.
    # Validation exists to fail at load, with the path in the message.
    "compute",
)

REQUIRED_PATH_KEYS = (
    "data_dir",
    "papers_dir",
    "index_dir",
    "cache_dir",
    "logs_dir",
    "prompts_dir",
)


class ConfigError(RuntimeError):
    """config.yaml is missing, malformed, or a requested key does not exist."""


class Section:
    """Read-only attribute view over one mapping in config.yaml.

    Unknown attributes raise ConfigError rather than returning None, so a mistyped
    tunable fails where it is used instead of silently disabling a feature.
    """

    def __init__(self, path: str, data: Dict[str, Any]) -> None:
        object.__setattr__(self, "_path", path)
        object.__setattr__(
            self,
            "_data",
            {
                key: Section(f"{path}.{key}", value) if isinstance(value, dict) else value
                for key, value in data.items()
            },
        )

    def __getattr__(self, key: str) -> Any:
        data: Dict[str, Any] = object.__getattribute__(self, "_data")
        if key not in data:
            path: str = object.__getattribute__(self, "_path")
            known = ", ".join(sorted(data)) or "<empty section>"
            raise ConfigError(f"no config key '{path}.{key}'; {path} has: {known}")
        return data[key]

    def __setattr__(self, key: str, value: Any) -> None:
        raise ConfigError(f"config is read-only; cannot set '{self._path}.{key}'")

    def __contains__(self, key: str) -> bool:
        return key in object.__getattribute__(self, "_data")

    def __iter__(self) -> Iterator[str]:
        return iter(object.__getattribute__(self, "_data"))

    def get(self, key: str, default: Any = None) -> Any:
        """Value for `key`, or `default` when it is absent. Use only for genuine optionals."""
        return object.__getattribute__(self, "_data").get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        """Plain-dict copy of this section, nested sections flattened back to dicts."""
        return {
            key: value.as_dict() if isinstance(value, Section) else value
            for key, value in object.__getattribute__(self, "_data").items()
        }

    def __repr__(self) -> str:
        path: str = object.__getattribute__(self, "_path")
        keys = ", ".join(sorted(object.__getattribute__(self, "_data")))
        return f"<Section {path}: {keys}>"


class Paths:
    """Absolute paths for every artefact named in docs/DATA_SCHEMA.md section 5.

    Filenames live here and nowhere else, so no module hardcodes 'manifest.json'.
    """

    def __init__(self, root: Path, paths_section: Section) -> None:
        self.root = root
        self.data = root / paths_section.data_dir
        self.papers = root / paths_section.papers_dir
        self.index = root / paths_section.index_dir
        self.cache = root / paths_section.cache_dir
        self.logs = root / paths_section.logs_dir
        self.prompts = root / paths_section.prompts_dir

        # index artefacts — manifest, chunks, vectors and the FAISS index itself
        self.manifest = self.index / "manifest.json"
        self.chunks = self.index / "chunks.jsonl"
        self.embeddings = self.index / "embeddings.npy"
        self.faiss_index = self.index / "faiss.index"

        # caches, all keyed by content hash rather than by path
        self.figures = self.cache / "figures"
        self.embedding_cache = self.cache / "embeddings.json"
        self.descriptions = self.cache / "descriptions.json"
        self.arxiv_queries = self.cache / "arxiv_queries.json"
        self.clusters = self.cache / "clusters.json"
        # Not a cache of results but of a refusal: when arXiv last rate-limited this IP.
        # Lives beside the caches because it is equally disposable — deleting it only
        # means the next call finds out from arXiv instead of from disk.
        self.arxiv_cooldown = self.cache / "arxiv_cooldown.json"

        # tool-call traces, one file per run
        self.traces = self.logs / "traces"

    def paper_pdf(self, paper_id: str) -> Path:
        """Where paper `paper_id`'s PDF lives on disk."""
        return self.papers / f"{paper_id}.pdf"

    def figure_dir(self, paper_id: str) -> Path:
        """Directory holding every extracted image for paper `paper_id`."""
        return self.figures / paper_id

    def figure_image(self, paper_id: str, figure_id: str) -> Path:
        """Where figure `figure_id`'s extracted image lives on disk."""
        return self.figure_dir(paper_id) / f"{figure_id}.png"

    def trace_file(self, run_id: str) -> Path:
        """The tool-call trace file for run `run_id`."""
        return self.traces / f"{run_id}.jsonl"

    def directories(self) -> List[Path]:
        """Every directory the system writes into."""
        return [self.data, self.papers, self.index, self.cache, self.figures, self.logs, self.traces]

    def ensure(self) -> None:
        """Create the on-disk scaffolding. Idempotent; safe to call at every entry point."""
        for directory in self.directories():
            directory.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"<Paths root={self.root}>"


class Config:
    """Parsed config.yaml plus environment secrets.

    Sections are reachable as attributes (`config.retrieval.k_retrieve`); `config.paths`
    is a `Paths` object rather than the raw YAML mapping it was built from.
    """

    def __init__(self, config_path: Path, env_path: Optional[Path] = None) -> None:
        if not config_path.is_file():
            raise ConfigError(f"config file not found: {config_path}")

        load_dotenv(env_path if env_path is not None else REPO_ROOT / ".env")

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConfigError(f"{config_path} did not parse to a mapping")

        missing = [name for name in REQUIRED_SECTIONS if name not in raw]
        if missing:
            raise ConfigError(f"{config_path} is missing sections: {', '.join(missing)}")

        missing_paths = [key for key in REQUIRED_PATH_KEYS if key not in raw["paths"]]
        if missing_paths:
            raise ConfigError(f"{config_path} paths section is missing: {', '.join(missing_paths)}")

        self.source_path = config_path
        self._raw = raw
        for name, value in raw.items():
            object.__setattr__(self, name, Section(name, value) if isinstance(value, dict) else value)

        # Overrides the raw `paths` Section set in the loop above.
        self.paths = Paths(REPO_ROOT, Section("paths", raw["paths"]))

    @property
    def api_key(self) -> Optional[str]:
        """The OpenRouter key from the environment, or None when it is not set."""
        key = os.environ.get(API_KEY_ENV_VAR, "").strip()
        return key or None

    def require_api_key(self) -> str:
        """The OpenRouter key, or a ConfigError naming exactly what to do about it."""
        key = self.api_key
        if not key:
            raise ConfigError(
                f"{API_KEY_ENV_VAR} is not set. Copy .env.example to .env and fill it in."
            )
        return key

    @property
    def app_name(self) -> str:
        """Value sent to OpenRouter as X-Title, for usage attribution."""
        return os.environ.get(APP_NAME_ENV_VAR, "").strip() or "sciagent"

    def model_for(self, role: str) -> str:
        """Model id for an LLM role: 'agent', 'vision' or 'utility'."""
        model = self.llm.get(f"{role}_model")
        if model is None:
            raise ConfigError(f"unknown llm role '{role}'; expected one of agent, vision, utility")
        if not model:
            raise ConfigError(f"llm.{role}_model is empty in {self.source_path}")
        return str(model)

    def prompt(self, name: str) -> str:
        """Contents of src/prompts/{name}.md. Prompts are files so they can be diffed."""
        prompt_path = self.paths.prompts / f"{name}.md"
        if not prompt_path.is_file():
            raise ConfigError(f"prompt not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")

    def as_dict(self) -> Dict[str, Any]:
        """Plain-dict copy of the parsed YAML, before path resolution."""
        return dict(self._raw)

    def __repr__(self) -> str:
        return f"<Config {self.source_path}>"


def config_path() -> Path:
    """The config file to load: $SCIAGENT_CONFIG if set, else config.yaml at the repo root."""
    override = os.environ.get(CONFIG_PATH_ENV_VAR, "").strip()
    return Path(override).expanduser().resolve() if override else REPO_ROOT / "config.yaml"


def load_config(path: Optional[Path] = None) -> Config:
    """Build a fresh Config. Prefer the module-level CFG; this exists for tests."""
    return Config(path if path is not None else config_path())


CFG: Config = load_config()
