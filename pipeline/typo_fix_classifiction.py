import math
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd
import typer

app = typer.Typer(add_completion=False)

# Hard boundaries (never merge across)
HARD_KEEP_WORDS_DEFAULT: Set[str] = {
    "OF", "AND", "OR", "TO", "WITH", "WITHOUT", "IN", "ON", "AT", "BY", "FOR", "FROM",
    "THE", "AS", "IS", "ARE", "WAS", "WERE", "BEEN", "BE", "DUE", "VIA",
    "PER", "NOS", "NEC", "NOT", "W/O", "W/"
}

# Soft boundaries: treat as boundaries only if they are truly standalone words
SOFT_KEEP_WORDS: Set[str] = {"A", "AN"}

# Known targets that we want to be able to merge even if the file rarely contains them “clean”
EXTRA_WORDS_DEFAULT: Set[str] = {
    "FENTANYL",
    "FLUOROFENTANYL",
    "METHAMPHETAMINE",
    "CARDIOVASCULAR",
    "EXACERBATION",
    "ETHANOL",
    "ALCOHOL",
    "ALCOHOLISM",
    "INTOXICATION",
    "TOXICITY",
}

PUNCT_BOUNDARY_RE = re.compile(r"[,\.;:/]$")


def token_core(tok: str) -> str:
    """Uppercase letters/digits only (drop punctuation)."""
    return re.sub(r"[^A-Z0-9]", "", tok.upper())


def trailing_punct(tok: str) -> str:
    """Return trailing punctuation (e.g., ',' ')' etc), if any."""
    m = re.match(r"^.*?([^\w]+)$", tok)
    return m.group(1) if m else ""


def build_vocab(df: pd.DataFrame, columns: List[str]) -> Counter:
    vocab = Counter()
    for col in columns:
        for val in df[col].dropna().astype(str):
            for tok in val.split():
                c = token_core(tok)
                if c:
                    vocab[c] += 1
    return vocab


def premerge_keepwords(tokens: List[str], keep_words: Set[str], max_parts: int = 4) -> List[str]:
    """
    Merge fragments that form a keep word (e.g., A ND -> AND, AN D -> AND).
    This prevents later logic from accidentally merging AND with neighbors.
    """
    out: List[str] = []
    i = 0
    while i < len(tokens):
        merged_tok: Optional[str] = None
        used_k = 0

        for k in range(2, max_parts + 1):
            if i + k > len(tokens):
                break
            cand_core = "".join(token_core(t) for t in tokens[i : i + k])
            if cand_core in keep_words:
                suffix = trailing_punct(tokens[i + k - 1])
                merged_tok = cand_core + suffix
                used_k = k
                break

        if merged_tok is not None:
            out.append(merged_tok)
            i += used_k
        else:
            out.append(tokens[i])
            i += 1

    return out


def looks_like_fragment_soft_keep(
    toks: List[str],
    idx: int,
    vocab: Counter,
    extra_words: Set[str],
    max_merge: int,
    min_freq: int,
) -> bool:
    """
    Decide whether 'A'/'AN' at toks[idx] is a fragment (part of a larger word),
    based on whether merging a nearby window yields a known/valid word.
    """
    if idx < 0 or idx >= len(toks):
        return False

    core = token_core(toks[idx])
    if core not in SOFT_KEEP_WORDS:
        return False

    # Try windows that include this token (start a bit before it)
    start = max(0, idx - 2)
    end = min(len(toks), idx + max_merge)

    for i in range(start, idx + 1):
        for j in range(idx + 1, min(len(toks), i + max_merge) + 1):
            window = toks[i:j]
            if not window or (idx < i or idx >= j):
                continue

            # Don't cross punctuation boundaries
            ok = True
            for k in range(i + 1, j):
                if PUNCT_BOUNDARY_RE.search(toks[k - 1]):
                    ok = False
                    break
            if not ok:
                continue

            merged = "".join(token_core(t) for t in window)
            if not merged:
                continue

            if merged in extra_words or vocab.get(merged, 0) >= min_freq:
                return True

    return False

def build_extra_words_from_patterns(substance_patterns: dict) -> set[str]:
    """
    Lift patterns from your old detection script into a lexicon of
    space-insensitive target words used for repairs.
    """
    extra = set()
    for label, patterns in substance_patterns.items():
        # include the label itself if it's a single token-ish (optional but helpful)
        if label:
            extra.add(token_core(label))

        for p in patterns or []:
            extra.add(token_core(p))  # token_core removes spaces/punct & uppercases
    # remove empties
    extra.discard("")
    return extra


def premerge_known_targets(tokens: list[str], extra_words: set[str], max_merge: int = 6) -> list[str]:
    """
    Greedily merge runs of tokens if their merged core is a known target word.
    This happens BEFORE keep-word logic, so it can fix HERO IN -> HEROIN.
    Prefer the longest match at each position.
    """
    out = []
    i = 0
    while i < len(tokens):
        best_j = None

        # Try longest-first for stability
        for j in range(min(len(tokens) - 1, i + max_merge - 1), i, -1):
            # don't cross punctuation boundary
            ok = True
            for k in range(i + 1, j + 1):
                if re.search(r"[,\.;:/]$", tokens[k - 1]):
                    ok = False
                    break
            if not ok:
                continue

            merged_core = "".join(token_core(t) for t in tokens[i : j + 1])
            if merged_core and merged_core in extra_words:
                best_j = j
                break

        if best_j is not None:
            out.append("".join(tokens[i : best_j + 1]))
            i = best_j + 1
        else:
            out.append(tokens[i])
            i += 1

    return out



def repair_cell(
    text: str,
    vocab: Counter,
    hard_keep: Set[str],
    extra_words: Set[str],
    max_merge: int = 6,
    min_freq: int = 2,
) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return text

    s = str(text).strip()
    if not s:
        return s

    # 1) split
    toks = s.split()

    # 2) merge known targets FIRST (fixes HERO IN, ETH AN OL, METH A MPHETAMINE, etc.)
    toks = premerge_known_targets(toks, extra_words=extra_words, max_merge=max_merge)

    # 3) then your existing keepword pre-merge (AND, OF, etc.)
    toks = premerge_keepwords(toks, keep_words=set(hard_keep), max_parts=4)

    out: List[str] = []
    i = 0

    while i < len(toks):
        tok = toks[i]
        core_i = token_core(tok)

        # HARD boundaries: never merge across
        if core_i in hard_keep:
            out.append(core_i + trailing_punct(tok))
            i += 1
            continue

        # SOFT boundaries: treat as boundary only if it doesn't look like a fragment
        if core_i in SOFT_KEEP_WORDS and not looks_like_fragment_soft_keep(
            toks, i, vocab, extra_words, max_merge=max_merge, min_freq=min_freq
        ):
            out.append(core_i + trailing_punct(tok))
            i += 1
            continue

        best_j: Optional[int] = None
        best_score = -1.0

        for j in range(i + 1, min(len(toks), i + max_merge)):
            if PUNCT_BOUNDARY_RE.search(toks[j - 1]):
                break

            window = toks[i : j + 1]

            # Never merge if any HARD keep word appears in the window
            if any(token_core(t) in hard_keep for t in window):
                continue

            merged_core = "".join(token_core(t) for t in window)
            if not merged_core:
                continue

            freq = vocab.get(merged_core, 0)
            in_extra = merged_core in extra_words

            if in_extra or freq >= min_freq:
                score = (math.log(freq + 1) * 10 if freq else 0) + len(merged_core) + (20 if in_extra else 0)
                if score > best_score:
                    best_score = score
                    best_j = j

        if best_j is not None:
            out.append("".join(toks[i : best_j + 1]))
            i = best_j + 1
            continue

        # Fallback merge for tiny fragments — but DON'T glue onto an already-complete known word
        if i + 1 < len(toks) and not PUNCT_BOUNDARY_RE.search(toks[i]):
            a, b = toks[i], toks[i + 1]
            ca, cb = token_core(a), token_core(b)

            if ca and cb and ca not in hard_keep and cb not in hard_keep:
                # Prevent ETHANOL+T => ETHANOLT, METHAMPHETAMINE+INT => METHAMPHETAMINEINT
                if ca in extra_words and len(cb) <= 3:
                    out.append(a)
                    i += 1
                    continue

                if (len(ca) <= 3 or len(cb) <= 3) and len(ca + cb) >= 4:
                    out.append(a + b)
                    i += 2
                    continue

        out.append(tok)
        i += 1

    res = " ".join(out)
    res = re.sub(r"\s+([,;:.])", r"\1", res)
    res = re.sub(r"\(\s+", "(", res)
    res = re.sub(r"\s+\)", ")", res)
    return res


@app.command()
def clean_csv(
    input_path: Path = typer.Argument("./pipeline_steps/input_files/raw/PRADMEC1121.csv", exists=True, dir_okay=False, readable=True),
    output_path: Path = typer.Argument("PRADMEC1121_cleaned.csv", dir_okay=False),
            columns: List[str] = typer.Option(
        None,
        "--column",
        "-c",
        help="Column(s) to clean. Repeat: -c CauseA -c CauseB. Default: all columns.",
    ),
    max_merge: int = typer.Option(6, help="Max number of adjacent tokens to consider merging."),
    min_freq: int = typer.Option(2, help="Min vocab frequency required to accept a merged word (unless in extra words)."),
    extra_word: List[str] = typer.Option(
        [],
        "--extra-word",
        help="Extra target words to merge (repeatable). Example: --extra-word NALOXONE",
    ),
):
    df = pd.read_csv(input_path, dtype=str, encoding_errors="ignore")
    cols = columns if columns else df.columns.tolist()

    substance_patterns = {
    "Fentanyl": ["FENTANYL", "FENTANLY", "FENTAN", "FETANYL"],
    "Heroin": ["HEROIN"],
    "Methamphetamine": ["METHAMPHETAMINE", "METAMPHETAMINE", "METH AMPHETAMINE",
                        "METAMPHETA", "METHAMPHETA"],
    "Cocaine": ["COCAINE", "COCAIN", "COCA"],
    "Prescription.opioids": ["OXYCODONE", "HYDROCODONE", "MORPHINE", "TRAMADOL",
                             "CODEINE", "HYDROMORPHONE", "OXYMORPHONE"],
    "Alcohol": ["ETHANOL", "ALCOHOL", "ETHANO", "ETHAN"],
    "Benzodiazepines": ["BENZODIAZEPINE", "DIAZEPAM", "ALPRAZOLAM", "LORAZEPAM",
                        "CLONAZEPAM", "TEMAZEPAM", "XANAX", "VALIUM"],
    "Others": []
}


    hard_keep = set(HARD_KEEP_WORDS_DEFAULT)
    extra_words = set(EXTRA_WORDS_DEFAULT) | build_extra_words_from_patterns(substance_patterns)

    vocab = build_vocab(df, cols)

    for col in cols:
        df[col] = df[col].apply(
            lambda x: repair_cell(
                x,
                vocab=vocab,
                hard_keep=hard_keep,
                extra_words=extra_words,
                max_merge=max_merge,
                min_freq=min_freq,
            )
        )

    df.to_csv(output_path, index=False)
    typer.echo(f"Wrote cleaned file: {output_path}")


if __name__ == "__main__":
    app()
