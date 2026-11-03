import hashlib
import json

from config import BUNDLED_MODELS


def test_bundled_models_match_their_recorded_source():
    source = json.loads((BUNDLED_MODELS / "SOURCE.json").read_text())
    found = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in BUNDLED_MODELS.glob("*.pkl")}
    assert found == source["sha256"]
