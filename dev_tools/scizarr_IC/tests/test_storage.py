"""Credential plumbing in storage_for and the explanations for open failures (no network)."""
from __future__ import annotations

import icechunk

from scizarr_ic import ScizarrError
from scizarr_ic.storage import explain_open_failure, storage_for


def test_remote_storage_uses_env_credentials_unless_anonymous(monkeypatch):
    seen: dict = {}

    def fake(**kw):
        seen.clear()
        seen.update(kw)

    monkeypatch.setattr(icechunk, "gcs_storage", fake)
    monkeypatch.setattr(icechunk, "s3_storage", fake)
    storage_for("gs://bucket/some/prefix")
    assert seen == {"bucket": "bucket", "prefix": "some/prefix", "from_env": True}
    storage_for("s3://bucket", anonymous=True)
    assert seen == {"bucket": "bucket", "prefix": None, "anonymous": True}


def test_open_failures_name_the_cause():
    missing = explain_open_failure("gs://b/p", icechunk.RepositoryNotFoundError("the repository doesn't exist"))
    creds = explain_open_failure(
        "gs://b/p", icechunk.StorageError("object store error: Error performing token request\ncontext:\n 0: ...")
    )
    s3 = explain_open_failure("s3://b/p", icechunk.StorageError("AccessDenied"))
    local = explain_open_failure("/tmp/x", icechunk.StorageError("boom\ncontext"))
    assert all(isinstance(e, ScizarrError) for e in (missing, creds, s3, local))
    assert str(missing) == "No icechunk repository at 'gs://b/p'"
    assert "gcloud auth application-default login" in str(creds) and "anonymous=True" in str(creds)
    assert "context" not in str(creds)                      # first line only, no Rust backtrace
    assert "aws sso login" in str(s3)
    assert str(local) == "Cannot open '/tmp/x': boom"
