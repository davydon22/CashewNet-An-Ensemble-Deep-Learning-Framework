"""
Checks whether the fold-level ensemble significance test,
scores 0.9901/0.9916/0.9893/0.9902/0.9857) used TTA when constructing its
per-fold ensemble evaluation -- this determines whether that section needs
to be recomputed under the new flips-only TTA scheme, or whether it's
already TTA-independent (raw ensemble-only) and unaffected.


Run from the project root:
    python3 check_fold_level_tta_usage.py
"""
import inspect
import re

try:
    from evaluate import fold_level_ensemble_macro_f1
    source = inspect.getsource(fold_level_ensemble_macro_f1)
    print("Found fold_level_ensemble_macro_f1 in evaluate.py. Source:")
    print("=" * 70)
    print(source)
    print("=" * 70)

    if re.search(r"use_tta\s*=\s*True", source):
        print("\nRESULT: use_tta=True found -- fold-level ensemble scores "
              "DO use TTA (the old 6-view scheme). This section needs to be recomputed "
              "under flips-only TTA for consistency with the new headline result.")
    elif re.search(r"use_tta\s*=\s*False", source):
        print("\nRESULT: use_tta=False found -- fold-level ensemble scores "
              "do NOT use TTA. This section is unaffected by the TTA scheme change and "
              "does not need to be recomputed.")
    elif "predict_batch" in source and "use_tta" not in source:
        print("\nRESULT: predict_batch is called WITHOUT an explicit use_tta argument -- "
              "check predict_batch's default value for use_tta directly (see below) to "
              "determine which behavior applies.")
        from evaluate import EnsembleModel
        sig = inspect.signature(EnsembleModel.predict_batch)
        print(f"predict_batch signature: {sig}")
        default = sig.parameters.get("use_tta")
        if default is not None:
            print(f"Default value of use_tta: {default.default}")
    else:
        print("\nRESULT: Could not determine TTA usage automatically from source inspection. "
              "Please read the printed source above manually to check whether test-time "
              "augmentation (multiple views averaged) is applied before this function "
              "computes its per-fold ensemble predictions.")

except ImportError as e:
    print(f"Could not import fold_level_ensemble_macro_f1 from evaluate.py: {e}")
    print("Check the actual function/module name used for the fold-level "
          "ensemble computation and adjust the import above accordingly.")
except Exception as e:
    print(f"Unexpected error during inspection: {e}")
    print("Fall back to manually reading evaluate.py's fold-level ensemble function.")
