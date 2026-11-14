"""Financial PhraseBank loader (sentiment exam; labels: negative / neutral / positive)."""

import zipfile

import pandas as pd

from src.config import ROOT, load_config


def load_phrasebank() -> pd.DataFrame:
    """Read the configured agreement file from the downloaded archive."""
    cfg = load_config()
    spec = cfg["evals"]["phrasebank"]
    archive = ROOT / cfg["paths"]["phrasebank_dir"] / spec["file"]
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.endswith(spec["member"]))
        lines = bundle.read(member).decode("latin-1").splitlines()
    pairs = [line.rsplit("@", 1) for line in lines if "@" in line]
    frame = pd.DataFrame(pairs, columns=["sentence", "label"])
    frame["sentence"] = frame["sentence"].str.strip()
    frame["label"] = frame["label"].str.strip()
    return frame


def sample_phrasebank() -> pd.DataFrame:
    """Draw the fixed-seed evaluation sample."""
    cfg = load_config()
    frame = load_phrasebank().sample(n=cfg["evals"]["phrasebank"]["n"], random_state=cfg["seed"])
    frame = frame.reset_index(drop=True)
    frame.insert(0, "id", range(1, len(frame) + 1))
    return frame
