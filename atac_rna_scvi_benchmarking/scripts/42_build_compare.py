"""Fill report/compare_run2_template.html with compare_run2.json + report/compare_run2_findings.html -> report/compare_run2.html."""
import json

from scvi_pca import config

R = config.REPO / "report"


def main():
    data = json.loads((R / "compare_run2.json").read_text())
    have = {r["ds"] for r in data["rows"]}
    rna = [d for d in ("mmrna", "rna13m") if d in have]
    atac = [d for d in ("mmatac", "covatac") if d in have]
    f = R / "compare_run2_findings.html"
    html = (R / "compare_run2_template.html").read_text()
    html = (html.replace("__DATA__", json.dumps(data)).replace("__RNA_DS__", json.dumps(rna))
            .replace("__ATAC_DS__", json.dumps(atac)).replace("__FINDINGS__", f.read_text() if f.exists() else ""))
    (R / "compare_run2.html").write_text(html)
    print(f"wrote {R / 'compare_run2.html'} ({len(html):,} bytes; rna={rna}, atac={atac})")


if __name__ == "__main__":
    main()
