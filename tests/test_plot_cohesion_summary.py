#!/usr/bin/env python3
"""Unit tests for scripts/plot_cohesion_summary.py's CSV reshaping and labels."""

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "plot_cohesion_summary",
    Path(__file__).resolve().parents[1] / "scripts" / "plot_cohesion_summary.py",
)
pcs = importlib.util.module_from_spec(_SPEC)
sys.modules["plot_cohesion_summary"] = pcs
_SPEC.loader.exec_module(pcs)

HEADER = (
    "density,densifier,tag,collapsing,method,cell,variant,n_bench,category,n,"
    "within,null_mean,z,p,A\n"
)


def test_payload_from_csv_carries_view_metadata(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text(
        HEADER
        + "0,C,pa,aggregate,,0|C|pa|s_aggregate,with_g,3,subject:math,2,0.2,0.4,1.0,0.01,0.5\n"
        + "0,C,pa,aggregate,knn,0|C|pa|s_aggregate|mknn,with_g,3,subject:math,2,0.2,0.4,1.0,0.01,0.5\n",
        encoding="utf-8",
    )
    payload = pcs.payload_from_csv(p)
    agg, per = payload["0|C|pa|s_aggregate"], payload["0|C|pa|s_aggregate|mknn"]
    assert (agg["density"], agg["densifier"], agg["tag"], agg["collapsing"]) == (
        "0", "C", "pa", "aggregate")
    assert "method" not in agg
    assert per["method"] == "knn"
    assert len(agg["benchmarks"]) == 3
    assert pcs.column_label(agg) == "C"
    assert pcs.column_label(per) == "C · knn"
    assert pcs.column_label({**per, "method": "softimpute"}) == "C · si"


def test_payload_from_csv_rejects_pre_metadata_csv(tmp_path):
    p = tmp_path / "old.csv"
    p.write_text("cell,variant,n_bench,category,n,within,null_mean,z,p,A\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="re-run"):
        pcs.payload_from_csv(p)
