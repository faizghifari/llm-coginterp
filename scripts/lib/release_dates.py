"""Derive `release_date` (YYYY-MM floor) for benchmarks and models.

Two-stage design. This module is stage one: everything derivable *from data
already in the tables*, at zero lookup cost. Stage two (external research per
row, via the hermes agent) should only ever see what this leaves behind.

Why month precision: a bare year is too coarse to order releases within a year,
which is the whole point of having the field. Where a full YYYY-MM-DD is already
recorded it is kept as-is -- more precision is never discarded.

Provenance is recorded alongside every value in `release_date_source`, because
a date derived from an arXiv identifier and a date researched by an agent are
not equally trustworthy and must stay distinguishable after the fact:

  existing    already present and valid; untouched
  arxiv_id    decoded from an arXiv identifier (YYMM.NNNNN -> 20YY-MM)
  name_stamp  decoded from a date stamp inside the model id/name/repo
  hermes      researched externally (written by stage two, not here)

The arXiv decoding is exact, not heuristic: arXiv identifiers since 2007 encode
submission year and month in their first four digits. Validated against the 117
benchmarks that carry both an arXiv URL and a `year` value -- 101 agree exactly,
15 differ by one year (preprint vs publication, expected), and 1 disagrees
further (`summarization_cnndm`, whose `year` of 2023 is simply wrong; the paper
is arXiv 1704.04368). Note this means the derived date is a *paper* date, which
for a benchmark is the best available proxy for release.
"""
import re

# arXiv identifiers: YYMM.NNNNN (2007-present). Also tolerate the older
# archive/YYMMNNN form's date portion where it appears in a URL.
_ARXIV = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{2})(\d{2})\.\d{4,5}", re.I)

# Date stamps embedded in model identifiers: 20240620, 2024-06-20, 2024_06,
# claude-3-5-sonnet-20240620, gpt-4-0613 is NOT matched (no year -> ambiguous).
_STAMP = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?\d{2}\b|(20\d{2})[-_](\d{2})\b")

# What a usable release_date looks like once normalised.
_VALID = re.compile(r"^(19|20)\d{2}-(0[1-9]|1[0-2])(-\d{2})?$")

BENCHMARK_URL_COLUMNS = ("paper_url", "source_url", "other_url", "github_url",
                         "hf_url", "huggingface_url")
MODEL_NAME_COLUMNS = ("model_id", "model_name", "hf_repo", "url")


def is_valid_release_date(value):
    """True only for YYYY-MM or YYYY-MM-DD -- i.e. month precision reached.

    This is the *precision* test, not the "should I keep this" test. A bare
    year is a legitimate stored value (the corroboration ladder below emits
    one whenever two passes agree on the year but not the month), it simply
    has not reached month precision yet. Use `is_preservable_release_date`
    for the keep-or-clear decision; conflating the two silently deleted
    year-precision values, see its docstring."""
    if value is None:
        return False
    return bool(_VALID.match(str(value).strip()))


def is_preservable_release_date(value):
    """True for anything worth keeping: month precision OR a bare year.

    Separate from `is_valid_release_date` because the two questions came
    apart. When this module was written the only non-`YYYY-MM` values in the
    column were junk -- a `128.0` context-window leak and float-formatted
    `2023.0` -- so "not month precision" and "not a date at all" were the
    same test. The corroboration ladder then started writing bare years on
    purpose (`corroborated_year`, 223 model + 163 benchmark rows), and the
    single shared test meant the next `enrich_*` run would have cleared every
    one of them as invalid. A bare year is coarse, not wrong.

    Still rejects `128.0`, `2023.0` and anything else non-date."""
    if value is None:
        return False
    v = str(value).strip()
    return bool(_VALID.match(v) or _YEAR_ONLY.match(v))


def from_arxiv_url(*values):
    """Decode YYYY-MM from the first arXiv identifier found. None if absent."""
    for v in values:
        if not isinstance(v, str):
            continue
        m = _ARXIV.search(v)
        if not m:
            continue
        yy, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            # arXiv IDs are 2007+, so a 2-digit year is unambiguously 20YY.
            return f"20{yy:02d}-{mo:02d}"
    return None


def from_name_stamp(*values):
    """Decode YYYY-MM from a date stamp inside an identifier. None if absent.

    Deliberately requires a 4-digit year: `gpt-4-0613` is a real checkpoint
    stamp but its year is implicit, and guessing it would be inventing data."""
    for v in values:
        if not isinstance(v, str):
            continue
        m = _STAMP.search(v)
        if not m:
            continue
        year = m.group(1) or m.group(3)
        month = m.group(2) or m.group(4)
        if year and month and 1 <= int(month) <= 12:
            return f"{year}-{int(month):02d}"
    return None


def _row_values(row, columns):
    return [row.get(c) for c in columns if c in row.index]


def enrich_benchmarks(benchmarks):
    """Return (benchmarks, report). Adds/fills `release_date` +
    `release_date_source` without overwriting valid existing values."""
    out = benchmarks.copy()
    if "release_date" not in out.columns:
        out["release_date"] = None
    dates, sources, cleared = [], [], []

    for _, row in out.iterrows():
        current = row.get("release_date")
        if is_preservable_release_date(current):
            dates.append(str(current).strip())
            # Keep whatever tier actually produced this value. Overwriting it
            # with "existing" would erase the provenance the column exists to
            # record -- `hf_createdat` and `single_haiku` are not equally
            # trustworthy and must stay distinguishable after the fact.
            prior = str(row.get("release_date_source") or "").strip()
            sources.append(prior if prior and prior != "nan" else "existing")
            continue
        if current is not None and str(current).strip() not in ("", "nan"):
            cleared.append((row.get("benchmark_id"), str(current).strip()))
        derived = from_arxiv_url(*_row_values(row, BENCHMARK_URL_COLUMNS))
        dates.append(derived)
        sources.append("arxiv_id" if derived else None)

    out["release_date"] = dates
    out["release_date_source"] = sources
    # "existing" counts every row that already carried a usable value,
    # whatever tier wrote it -- not just the literal "existing" label, which
    # now only marks values with no recorded provenance at all.
    kept = sum(1 for s in sources if s is not None and s != "arxiv_id")
    return out, {
        "total": len(out),
        "existing": kept,
        "arxiv_id": sources.count("arxiv_id"),
        "missing": sum(1 for s in sources if s is None),
        "month_precision": sum(1 for d in dates if is_valid_release_date(d)),
        "year_only": sum(1 for d in dates
                         if d and not is_valid_release_date(d)
                         and is_preservable_release_date(d)),
        "cleared_invalid": cleared,
    }


def enrich_models(models):
    """Return (models, report). Same contract as enrich_benchmarks."""
    out = models.copy()
    if "release_date" not in out.columns:
        out["release_date"] = None
    dates, sources, cleared = [], [], []

    for _, row in out.iterrows():
        current = row.get("release_date")
        if is_preservable_release_date(current):
            dates.append(str(current).strip())
            # Keep whatever tier actually produced this value. Overwriting it
            # with "existing" would erase the provenance the column exists to
            # record -- `hf_createdat` and `single_haiku` are not equally
            # trustworthy and must stay distinguishable after the fact.
            prior = str(row.get("release_date_source") or "").strip()
            sources.append(prior if prior and prior != "nan" else "existing")
            continue
        if current is not None and str(current).strip() not in ("", "nan"):
            cleared.append((row.get("model_id"), str(current).strip()))
        derived = from_name_stamp(*_row_values(row, MODEL_NAME_COLUMNS))
        dates.append(derived)
        sources.append("name_stamp" if derived else None)

    out["release_date"] = dates
    out["release_date_source"] = sources
    # "existing" counts every row that already carried a usable value,
    # whatever tier wrote it -- not just the literal "existing" label, which
    # now only marks values with no recorded provenance at all.
    kept = sum(1 for s in sources if s is not None and s != "name_stamp")
    return out, {
        "total": len(out),
        "existing": kept,
        "name_stamp": sources.count("name_stamp"),
        "missing": sum(1 for s in sources if s is None),
        "month_precision": sum(1 for d in dates if is_valid_release_date(d)),
        "year_only": sum(1 for d in dates
                         if d and not is_valid_release_date(d)
                         and is_preservable_release_date(d)),
        "cleared_invalid": cleared,
    }


# ── Stage two: two-pass agreement filter ────────────────────────────────────
#
# A single pass from the local model is ~70% accurate on release dates and
# carries no signal marking which rows are wrong (measured 2026-08-26 on a
# 10-model pilot: DeepSeek-V3 off by 7 months, Zephyr-7B-beta by 3, Phi-3-mini
# by 1). Writing that blind would make the column look authoritative while
# being wrong on roughly a third of rows -- worse than leaving it empty.
#
# So each row is researched TWICE under different framings, and a value is
# only accepted where the two passes corroborate each other. Agreement is not
# proof of correctness (both passes share a model, so they can be wrong
# together), but disagreement is strong evidence of unreliability, and it
# costs nothing to run locally.
#
# Acceptance ladder, loosest rung that both passes support:
#   same YYYY-MM        -> month precision, hermes_agreed_month
#   same YYYY, diff MM  -> year precision,  hermes_agreed_year
#   one YYYY-MM + one YYYY, same year -> year precision
#   different years / either UNKNOWN   -> REJECTED, left empty for review

_YEAR_ONLY = re.compile(r"^(19|20)\d{2}$")


def _norm_answer(value):
    """Normalise one pass's answer to ('month', 'YYYY-MM') | ('year', 'YYYY')
    | None. Anything unparseable is treated as no answer, deliberately -- a
    malformed reply is not evidence of anything."""
    if value is None:
        return None
    v = str(value).strip()
    if not v or v.upper() == "UNKNOWN":
        return None
    if _VALID.match(v):
        return ("month", v[:7])
    if _YEAR_ONLY.match(v):
        return ("year", v)
    return None


def reconcile(answer_a, answer_b):
    """Apply the acceptance ladder to two passes. Returns
    (value, source, reason) with value None when rejected."""
    a, b = _norm_answer(answer_a), _norm_answer(answer_b)
    if a is None or b is None:
        missing = "both passes" if (a is None and b is None) else "one pass"
        return None, None, f"no usable answer from {missing}"
    ya, yb = a[1][:4], b[1][:4]
    if ya != yb:
        return None, None, f"year disagreement ({a[1]} vs {b[1]})"
    if a[0] == "month" and b[0] == "month":
        if a[1] == b[1]:
            return a[1], "hermes_agreed_month", "exact agreement"
        return ya, "hermes_agreed_year", f"month disagreement ({a[1]} vs {b[1]})"
    # at least one pass only knew the year, and the years match
    return ya, "hermes_agreed_year", "year-precision agreement"


def read_answer_csv(path, with_evidence=False):
    """Read a `key,answer[,evidence]` file written by an agent.

    Two failure modes have to be tolerated, and they pull in opposite
    directions, so a proper CSV parse is tried FIRST and the lenient path is
    only a fallback:

    - Some agents do not quote keys containing commas (35 of our model_ids do,
      e.g. `ChatGPT (gpt-3.5-turbo, few-shot)`). Strict parsing rejects those.
    - Others quote correctly, including evidence text that contains commas
      (`"Stanford HELM Audio benchmark, no standalone paper found"`). Naive
      right-splitting tears those rows apart -- it silently produced a key of
      `parade,UNKNOWN` with the evidence as the answer.

    So: parse with the csv module; if a row does not yield the expected field
    count, fall back to peeling fields off the right for that row only.
    Returns {key: answer}, or {key: (answer, evidence)} when with_evidence.
    """
    import csv as _csv

    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(_csv.reader(fh))
    if not rows:
        return out
    header = [c.strip().lower() for c in rows[0]]
    ncols = 3 if len(header) >= 3 else 2
    start = 1 if header and header[0] == "key" else 0

    for row in rows[start:]:
        if not row or not any(c.strip() for c in row):
            continue
        if len(row) == ncols:
            key, answer = row[0], row[1]
            evidence = row[2] if ncols == 3 else ""
        else:
            # malformed: unquoted commas in the key. answer/evidence never
            # contain commas, so peel them off the right.
            line = ",".join(row)
            evidence = ""
            rest = line
            if ncols == 3:
                if "," not in rest:
                    continue          # truncated row (e.g. read mid-write)
                rest, evidence = rest.rsplit(",", 1)
            if "," not in rest:
                continue
            key, answer = rest.rsplit(",", 1)
        key = key.strip().strip('"')
        answer = answer.strip().strip('"')
        out[key] = (answer, evidence.strip().strip('"')) if with_evidence else answer
    return out
