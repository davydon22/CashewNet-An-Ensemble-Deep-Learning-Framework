"""
Analyzes completed domain-consistency judgments from sample_for_domain_audit.py,
computing the estimated rate of domain-mismatched (mislabeled) images per
class and overall, with a Wilson score confidence interval -- more reliable
than a normal-approximation interval when the true rate is low or the
sample size is modest, both of which are expected to apply here.

Run from the project root, after completing judgments_completed.csv:
    python3 analyze_domain_audit.py
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import norm

JUDGMENTS_PATH = "domain_audit_review/judgments_completed.csv"
CONFIDENCE = 0.95


def wilson_ci(successes, n, confidence=0.95):
    """Wilson score interval for a binomial proportion. More reliable than
    the normal approximation (p +/- z*SE) at small n or when p is near 0
    or 1, both plausible here given mislabeling is expected to be rare."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = p_hat + z**2 / (2 * n)
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return max(0.0, lo), min(1.0, hi)


def main():
    if not os.path.exists(JUDGMENTS_PATH):
        print(f"'{JUDGMENTS_PATH}' does not exist yet.")
        print()
        print("This means the manual review step hasn't happened yet -- running")
        print("sample_for_domain_audit.py only prepares the images and an empty")
        print("template; it does not judge them for you. To proceed:")
        print()
        print("  1. Open the images in domain_audit_review/ (sorted by filename,")
        print("     grouped by class) in a file explorer or image viewer.")
        print("  2. Open domain_audit_review/judgments_template.csv.")
        print("  3. For each row, fill in the 'domain_consistent' column with")
        print("     YES (genuinely shows leaf tissue matching the labeled class)")
        print("     or NO (shows something else -- wrong plant part, wrong")
        print("     species, or a domain mismatch like the cashew fruit photo.")
        print("  4. Save the completed file as domain_audit_review/judgments_completed.csv")
        print("     (a new file -- do not overwrite the template).")
        print("  5. Rerun this script.")
        return

    df = pd.read_csv(JUDGMENTS_PATH)

    unfilled = df["domain_consistent"].isna() | (df["domain_consistent"].astype(str).str.strip() == "")
    if unfilled.any():
        print(f"WARNING: {unfilled.sum()} row(s) have no judgment filled in yet. "
              f"These will be excluded from the analysis below -- fill them in for a complete result.")
        df = df[~unfilled]

    df["domain_consistent"] = df["domain_consistent"].astype(str).str.strip().str.upper()
    df["is_mismatch"] = df["domain_consistent"] == "NO"

    print(f"Total images reviewed: {len(df)}")
    print(f"Overall mismatches found: {df['is_mismatch'].sum()}")
    overall_lo, overall_hi = wilson_ci(df["is_mismatch"].sum(), len(df), CONFIDENCE)
    overall_rate = df["is_mismatch"].mean()
    print(f"Overall estimated mismatch rate: {overall_rate*100:.2f}% "
          f"(95% CI: {overall_lo*100:.2f}\u2013{overall_hi*100:.2f}%)")
    print()

    print("Per-class breakdown:")
    summary_rows = []
    for cls in sorted(df["labeled_class"].unique()):
        cls_df = df[df["labeled_class"] == cls]
        n = len(cls_df)
        mismatches = cls_df["is_mismatch"].sum()
        rate = mismatches / n if n > 0 else float("nan")
        lo, hi = wilson_ci(mismatches, n, CONFIDENCE)
        print(f"  {cls:<15} n={n:<4} mismatches={mismatches:<3} "
              f"rate={rate*100:5.2f}%  95% CI [{lo*100:5.2f}%, {hi*100:5.2f}%]")
        summary_rows.append({"class": cls, "n": n, "mismatches": mismatches,
                              "rate_pct": rate * 100, "ci_lo_pct": lo * 100, "ci_hi_pct": hi * 100})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("domain_audit_review/audit_summary.csv", index=False)
    print("\nSaved domain_audit_review/audit_summary.csv")

    if df["is_mismatch"].any():
        print("\nFlagged mismatches (for manual double-check before):")
        print(df[df["is_mismatch"]][["review_filename", "original_path", "labeled_class", "notes"]].to_string(index=False))


if __name__ == "__main__":
    main()
