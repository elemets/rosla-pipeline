#!/usr/bin/env python
"""Classification-only entry point: a string, a list of strings, a CSV/XLSX
path, or a DataFrame goes in; the BERT + NFLIS-regex drug labels come out.

    from classify_only import classify
    classify("ACUTE FENTANYL AND METHAMPHETAMINE TOXICITY")
    classify("records.csv", output="classified.csv")
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.classify import (
    apply_regex_classifier,
    create_text_col,
    drug_cols,
    predict,
    resolve_model_dir,
)
from pipeline.regex_classifier import DEFAULT_NFLIS, SEARCH_FIELDS

DEFAULT_MODEL = os.environ.get("ROC_MODEL", "bert_models/bioclinicalbert")

LABEL_COLS = drug_cols + ["Number_Substances", "Polysubstance"]


def _to_frame(source) -> pd.DataFrame:
    """Normalise any supported input into a DataFrame the classifier accepts."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, (list, tuple, pd.Series)):
        df = pd.DataFrame({"CauseA": [str(t) for t in source]})
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() in {".csv", ".xlsx", ".xls"} and path.exists():
            df = pd.read_excel(path) if path.suffix.lower() != ".csv" else pd.read_csv(path)
        else:
            df = pd.DataFrame({"CauseA": [str(source)]})
    else:
        raise TypeError(f"Unsupported input type: {type(source)!r}")

    df.columns = df.columns.str.replace(" ", "")

    # A file carrying a single free-text column is fed in as CauseA so both the
    # BERT text builder and the regex field search see it.
    if not any(col in df.columns for col in SEARCH_FIELDS):
        if "text" in df.columns:
            df["CauseA"] = df["text"]
        else:
            raise ValueError(
                "Input has no recognised text column. Expected one of "
                f"{SEARCH_FIELDS} or 'text'."
            )
    return df


def classify(
    source,
    model: str = DEFAULT_MODEL,
    use_regex: bool = True,
    output=None,
    batch_size: int = 64,
    nflis_path=DEFAULT_NFLIS,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the BERT classifier and the NFLIS regex layer over `source`.

    source     text string, list of strings, .csv/.xlsx path, or DataFrame
    model      checkpoint name under models/, or an explicit path
    use_regex  apply the NFLIS regex supplement and MDMA correction
    output     optional CSV path to write the classified records to
    """
    model_dir = resolve_model_dir(model)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_dir}. Set ROC_MODEL or pass "
            "model= with the checkpoint name under models/ or a full path."
        )

    df = create_text_col(_to_frame(source))
    # Single device on purpose: this entry point favours running anywhere over
    # squeezing throughput out of a multi-GPU host.
    result = predict(df, model, batch_size=batch_size, multi_gpu=False)
    if use_regex:
        result = apply_regex_classifier(result, nflis_path=nflis_path, verbose=verbose)

    if output:
        result.to_csv(output, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Classify overdose text with BERT + the NFLIS regex layer."
    )
    parser.add_argument("text", nargs="?", help="Free text to classify.")
    parser.add_argument("-i", "--input", help="Input CSV/XLSX file.")
    parser.add_argument("-o", "--output", help="Output CSV file.")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Model name or path.")
    parser.add_argument("--skip-regex", action="store_true", help="BERT only, no regex layer.")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if not args.text and not args.input:
        parser.error("give some text, or -i/--input with a CSV/XLSX file")

    result = classify(
        args.input or args.text,
        model=args.model,
        use_regex=not args.skip_regex,
        output=args.output,
        batch_size=args.batch_size,
    )

    if args.output:
        print(f"Wrote {len(result):,} classified records to {args.output}")
    else:
        cols = [c for c in LABEL_COLS if c in result.columns]
        print(result[["text"] + cols].to_string(index=False))


if __name__ == "__main__":
    main()
