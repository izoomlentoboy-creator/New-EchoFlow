"""Helpers to fetch & arrange a REAL voice-pathology corpus.

The container this was built in cannot reach the dataset hosts (network
policy), so run this on a machine with internet access.  Two well-known,
freely-available corpora are supported:

1. VOICED  (PhysioNet) - https://physionet.org/content/voiced/1.0.0/
   208 recordings: healthy + hyperkinetic / hypokinetic / reflux dysphonia.

2. Saarbruecken Voice Database (SVD) - http://stimmdb.coli.uni-saarland.de/
   2000+ speakers across many pathologies, with sustained vowels.

After downloading, arrange the audio into class subfolders matching
``echoflow.config.CLASSES``::

    data/real/
      healthy/        *.wav
      hyperfunctional/*.wav
      paralysis/      *.wav
      neurological/   *.wav
      inflammatory/   *.wav

then train with::

    python -m echoflow.train --data-root data/real

Below is a mapping from common SVD/VOICED diagnosis labels onto our 5-class
taxonomy.  Edit ``DIAGNOSIS_MAP`` to taste.
"""
from __future__ import annotations

import os
import shutil

# Map raw corpus diagnosis strings (lowercased) -> EchoFlow class.
DIAGNOSIS_MAP = {
    # healthy
    "healthy": "healthy", "normal": "healthy", "gesund": "healthy",
    # hyperfunctional: nodules, polyps, Reinke, hyperfunctional dysphonia
    "vocal fold nodules": "hyperfunctional", "polyp": "hyperfunctional",
    "reinke": "hyperfunctional", "hyperfunktionelle dysphonie": "hyperfunctional",
    "hyperkinetic dysphonia": "hyperfunctional",
    # paralysis / glottic insufficiency
    "recurrensparese": "paralysis", "vocal fold paralysis": "paralysis",
    "paralysis": "paralysis", "hypokinetic dysphonia": "paralysis",
    "psychogene dysphonie": "paralysis",
    # neurological
    "spasmodic dysphonia": "neurological", "spasmodische dysphonie": "neurological",
    "parkinson": "neurological", "dystonie": "neurological",
    # inflammatory
    "laryngitis": "inflammatory", "reflux laryngitis": "inflammatory",
    "oedem": "inflammatory", "edema": "inflammatory",
}


def map_diagnosis(raw: str) -> str | None:
    raw = (raw or "").strip().lower()
    if raw in DIAGNOSIS_MAP:
        return DIAGNOSIS_MAP[raw]
    for key, cls in DIAGNOSIS_MAP.items():
        if key in raw:
            return cls
    return None


def arrange(src_files_with_labels, out_root: str):
    """Copy (path, raw_diagnosis) pairs into class subfolders.

    ``src_files_with_labels``: iterable of (audio_path, raw_diagnosis_str).
    Unmapped diagnoses are skipped (and reported).
    """
    skipped = 0
    for path, raw in src_files_with_labels:
        cls = map_diagnosis(raw)
        if cls is None:
            skipped += 1
            continue
        dst_dir = os.path.join(out_root, cls)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(path, os.path.join(dst_dir, os.path.basename(path)))
    print(f"Arranged corpus under {out_root} (skipped {skipped} unmapped files)")


if __name__ == "__main__":
    print(__doc__)
