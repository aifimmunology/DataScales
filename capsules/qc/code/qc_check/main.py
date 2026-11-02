"""QC review of an AnnData store, and, with `apply`, the curated copy of its passing cells.

Every run reads a few obs columns, computes per-sample metrics, flags and a downsampled embedding, settles the
decisions, and writes the report. Only `apply`, with a previewed and unchanged decisions file, writes cells.
"""
from __future__ import annotations

import json
import logging
import platform
import resource
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from annizarr.errors import AnzError

import ambient
import decisions as dec
import embed
import metrics
import report
import samples
import write
from celltypes import prob_column
from config import ConfigError, Settings, load_settings
from profiles import DOUBLETS, SCRNA, SPATIAL, Profile
from source import SAMPLE, Source, SourceError, open_source

logger = logging.getLogger("qc")

PACKAGES = ("annizarr", "anndata", "icechunk", "zarr", "scanpy", "numpy", "pandas", "pyarrow", "matplotlib")


USER_ERRORS = (ConfigError, SourceError, metrics.MetricsError, dec.DecisionsError, write.ApplyError,
               AnzError)


def read_cells(source: Source, settings: Settings, profile: Profile,
               warnings: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """The obs columns this run needs, read one by one, with the sample column as `sample_id`.

    Missing metrics come from X when allowed, or always for profiles where that is cheap."""
    present = source.obs_columns()
    label = settings.label_column if settings.label_column in present else None
    rule_columns = [r.column for r in profile.rules]
    wanted = [source.sample_column, *profile.required, *rule_columns, DOUBLETS, *settings.groups(), *samples.REPORTED]
    if label:
        wanted += [label, prob_column(label)]
    columns = list(dict.fromkeys(c for c in wanted if c in present))
    categorical = (source.sample_column, *settings.groups(), *samples.REPORTED, label or "")
    obs = source.read_obs(columns, categorical=categorical).rename(columns={source.sample_column: SAMPLE})
    for rule in profile.rules:
        if rule.column not in obs and rule.column not in profile.required:
            warnings.append(f"No {rule.column} column, so the {rule.name} rule is off.")
    if source.sample_column != SAMPLE:
        warnings.append(f"No sample_id column, so samples are the table's regions ({source.sample_column}); "
                        "add a sample_id to review slide by slide.")
    lacking = metrics.missing(obs, profile.required)
    if not lacking:
        return obs, []
    if not (settings.compute_metrics or profile.compute_metrics):
        raise metrics.MetricsError(f"Store lacks QC metrics ({', '.join(lacking)} absent or NA for some cells); "
                                   "rerun with 'compute missing metrics' (reads all of X, about 16 KB per cell).")
    logger.info("Computing %s from X", ", ".join(lacking))
    computed = metrics.compute(source.root, source.var_names(), settings.band_cells)
    for column in lacking:
        obs[column] = computed[metrics.FROM_X[column]].to_numpy()
    warnings.append(f"QC metrics computed from X: {', '.join(lacking)} (absent or NA in the store).")
    return obs, lacking


def usable_groups(obs: pd.DataFrame, requested: list[str], warnings: list[str]) -> list[str]:
    groups = []
    for column in requested:
        if column not in obs:
            warnings.append(f"Group column {column} is absent; skipped.")
        elif obs[column].nunique(dropna=True) < 2:
            warnings.append(f"Group column {column} has a single value; skipped.")
        else:
            groups.append(column)
    return groups


def label_for_celltypes(obs: pd.DataFrame, settings: Settings, warnings: list[str]) -> str | None:
    if not settings.celltype_qc:
        return None
    if settings.label_column not in obs:
        warnings.append(f"No `{settings.label_column}` column; per-cell-type QC skipped. "
                        "Re-ingest with CellTypist on, or set a label column.")
        return None
    return settings.label_column


def report_decisions(source: Source, profile: Profile, file: dict | None, label: str | None,
                     sample_ids: list[str], flagged: set[str], curation: dict | None) -> dict:
    """defaults → the curation branch's decisions → the given file, then new samples settled, then sealed."""
    decisions = dec.empty(profile)
    if curation:
        decisions = dec.merge(decisions, curation["decisions"])
    if file:
        decisions = dec.merge(decisions, file)
    decisions["version"] = dec.VERSION
    decisions["input"] = {"kind": source.kind, "location": source.location, "fingerprint": source.fingerprint}
    decisions["label_column"] = label
    return dec.seal(dec.settle_new_samples(decisions, sample_ids, flagged))


def curation_line(source: Source, branch: str, curation: dict | None, decided: dict, sample_ids: list[str]) -> str:
    new = [s for s in sample_ids if s not in (curation["decisions"]["samples"] if curation else {})]
    pending = decided["pending"]
    tail = f"{len(new)} new samples ({len(pending)} flagged" + (f", pending: {', '.join(pending)})" if pending else ")")
    if source.kind != "icechunk":
        return f"Plain zarr input, so no curation branch; {len(sample_ids)} samples, pending: {', '.join(pending) or 'none'}."
    if curation is None:
        return f"No curation on '{branch}' yet; {tail}."
    covered = len(sample_ids) - len(new)
    created = str(curation.get("created", ""))[:10]
    return (f"{branch} covers {covered} of {len(sample_ids)} samples on main "
            f"(curated {created} from {str(curation.get('source', ''))[:8]}); {tail}.")


def decision_warnings(decided: dict, sample_ids: list[str]) -> list[str]:
    out = []
    if decided["cell_types"] and not decided.get("label_column"):
        out.append("cell_types overrides are ignored: per-cell-type QC is off or the label column is absent.")
    unknown = sorted(set(decided["samples"]) - set(sample_ids))
    if unknown:
        out.append(f"Decisions name samples not in the input: {', '.join(unknown)}.")
    return out


def embedding_frame(emb: pd.DataFrame, rows: np.ndarray, obs: pd.DataFrame, reason: pd.Categorical,
                    groups: list[str], label: str | None) -> pd.DataFrame:
    frame = emb.assign(row=rows, sample_id=obs["sample_id"].to_numpy()[rows])
    frame["qc_fail_reason"] = pd.Series(reason[rows]).astype(object).fillna("pass").to_numpy()
    for column in groups + ([label] if label else []):
        frame[column] = obs[column].to_numpy()[rows]
    return frame


def log_warnings(warnings: list[str]) -> list[str]:
    for warning in warnings:
        logger.warning("%s", warning)
    return warnings


def for_apply(settings: Settings, file: dict) -> Settings:
    """An apply run takes its per-cell-type QC and label column from the decisions file, not the panel."""
    label = file.get("label_column")
    return replace(settings, celltype_qc=bool(label), label_column=label or settings.label_column)


def flagged_samples(source: Source, settings: Settings, profile: Profile, obs: pd.DataFrame, var_names: np.ndarray,
                    label_column: str | None) -> tuple[pd.DataFrame, embed.Sampled, ambient.Ambient]:
    """The per-sample table with its flags, plus the sampled cells the ambient indicator and embedding share."""
    table = samples.sample_metrics(obs, profile.metrics, label_column)
    logger.info("%d cells in %d samples; sampling cells for the embedding", len(obs), len(table))
    picked = embed.sample_cells(source.root, obs[SAMPLE], settings.embed_per_sample or profile.embed_per_sample,
                                settings.embed_max_cells, settings.seed)
    amb = ambient.indicator(picked.counts, var_names, obs[SAMPLE].to_numpy()[picked.rows],
                            obs[label_column].to_numpy()[picked.rows] if label_column else None,
                            dec.doublet_calls(obs)[picked.rows], settings.label_column) if profile.ambient \
        else ambient.Ambient(pd.DataFrame(), {})
    table = table.join(amb.table)
    table["flags"] = samples.flag(table, amb.flags)
    return table, picked, amb


def build_review(source: Source, settings: Settings, profile: Profile, file: dict | None) -> report.Review:
    warnings: list[str] = []
    var_names = source.var_names()
    if profile.mito and not metrics.mito_mask(var_names).any():
        warnings.append(f"var has no genes starting with {metrics.MITO_PREFIX}; mito % will be 0 for every cell.")
    obs, computed = read_cells(source, settings, profile, warnings)
    groups = usable_groups(obs, settings.groups(), warnings)
    label_column = settings.label_column if settings.label_column in obs else None
    table, picked, amb = flagged_samples(source, settings, profile, obs, var_names, label_column)
    warnings += amb.warnings
    sample_ids = [str(s) for s in table.index]

    label = label_for_celltypes(obs, settings, warnings)
    curation = source.curation(settings.branch())
    if curation:
        curation["decisions"] = dec.validate(json.loads(curation["decisions"]), profile)
    if settings.apply:
        decided = file
    else:
        decided = report_decisions(source, profile, file, label, sample_ids, set(table.index[table["flags"].map(bool)]),
                                   curation)
    warnings += decision_warnings(decided, sample_ids)

    fails = dec.failures(obs, decided, profile)
    reason = dec.first_reason(fails)
    table = table.join(samples.fail_table(obs, fails))
    table["status"] = samples.status(table.index, decided)

    logger.info("Embedding %d sampled cells", len(picked.rows))
    emb = embed.embed(picked.counts, var_names, settings.seed)
    if emb is None:
        warnings.append(f"Embedding skipped: fewer than {embed.MIN_CELLS} sampled cells.")
    else:
        emb = embedding_frame(emb, picked.rows, obs, reason, groups, label_column)
    return report.Review(source=source, profile=profile, obs=obs, decisions=decided, fails=fails, reason=reason, table=table,
                         groups=groups, label=label_column, celltype_qc=bool(decided.get("label_column")),
                         computed_metrics=computed, ambient=amb, embedding=emb,
                         curation_line=curation_line(source, settings.branch(), curation, decided, sample_ids),
                         coords=source.coords(), warnings=log_warnings(warnings))


def write_run_json(results: Path, settings: Settings, review: report.Review, started: str,
                   applied: dict | None) -> None:
    source = review.source
    run = {
        "mode": "apply" if settings.apply else "report",
        "parameters": asdict(settings),
        "input": {"kind": source.kind, "location": source.location, "table": source.table,
                  "profile": review.profile.name, "fingerprint": source.fingerprint},
        "versions": {"python": platform.python_version(), **{p: version(p) for p in PACKAGES}},
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "peak_memory_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2),
    }
    run.update(n_cells=len(review.obs), n_pass=int(review.passed.sum()), n_samples=len(review.table),
               decisions_sha256=review.decisions[dec.HASH_KEY], pending=review.decisions["pending"])
    if applied:
        run["applied"] = applied
    (results / "run.json").write_text(json.dumps(run, indent=2, default=str))


def apply_summary(review: report.Review) -> pd.DataFrame:
    reason = pd.Series(review.reason, index=review.obs.index)
    removed = pd.crosstab(review.obs["sample_id"], reason).reindex(
        index=review.table.index, columns=review.rules, fill_value=0).add_prefix("removed_")
    table = pd.DataFrame({"before": review.table["n_cells"], "after": review.table["n_pass"]}).join(removed)
    return table.fillna(0).astype(np.int64).rename_axis("sample_id")


def run(settings: Settings) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    results = Path(settings.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    source = open_source(settings.input, settings.origin, settings.table)
    profile = SPATIAL if source.spatial else SCRNA
    logger.info("Input %s (%s%s, %s QC, fingerprint %s)", source.location, source.kind,
                f", SpatialData {source.table}" if source.spatial else "", profile.name, source.fingerprint)
    file = dec.load(settings.decisions, profile) if settings.decisions else None
    if settings.apply:
        write.preflight(file, source, settings.branch(), settings.target, settings.results_dir)
        settings = for_apply(settings, file)

    review = build_review(source, settings, profile, file)
    report.write_tables(review, results)
    logger.info("Writing plots")
    report.write_plots(review, results)
    applied = None
    if settings.apply:
        logger.info("Writing %d passing cells", int(review.passed.sum()))
        applied = write.apply(review, settings.branch(), settings.target, settings.reviewer, settings.results_dir,
                              report.cell_qc_digest(review))
        apply_summary(review).to_csv(results / "apply_summary.csv")
    text = report.write_summary(review, results, settings.apply, applied)
    write_run_json(results, settings, review, started, applied)
    logger.info("Done.\n%s", text)
    return {"review": review, "applied": applied}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    sc.settings.verbosity = 0
    try:
        run(load_settings(argv))
    except USER_ERRORS as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
