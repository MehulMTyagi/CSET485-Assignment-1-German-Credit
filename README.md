# CSET485 Assignment 1: German Credit

Reproducible analysis notebook for the German Credit route (roll suffix 044; random seed 44). The notebook covers preprocessing, linear and logistic regression, subgroup coefficient comparisons, an omitted-variable sensitivity check, confident prediction errors, and threshold-based fairness metrics.

## Run

Use Python 3.12. From this repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python analysis.py
```

Or open `CSET485_044_German_Credit.ipynb` in Jupyter and run every cell. The notebook and script regenerate analysis tables and charts under `results/`. If the local data file is missing, `analysis.py` downloads it from UCI.

## Data and attribution

Hans Hofmann (1994), *Statlog (German Credit Data)*, UCI Machine Learning Repository, DOI: https://doi.org/10.24432/C5NC77. Licensed CC BY 4.0. The data files and original attribute codebook are in `data/`.

UCI labels good and bad credit risk, not actual approve/reject decisions. The analysis maps good risk to the assignment's positive approval proxy.
