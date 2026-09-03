"""Shared chart styling: palette, rcParams, title/subtitle/caption helpers,
and human-readable labels for the model's raw pipeline column names.

Every figure in reports/figures/ is meant to be read without the code behind
it -- a finding-first title, a plain-language subtitle giving context, and
(where the reader needs it) a caption explaining how to read the chart.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- palette (validated categorical + sequential ramp; see dataviz skill) ---
BLUE = "#2a78d6"      # slot 1 -- "stay" / low risk
ORANGE = "#eb6834"    # slot 2 -- "churn" / high risk
RED = "#e34948"
GREEN = "#1baf7a"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Diverging pair (blue <-> red through a neutral gray midpoint) -- for data
# that spans positive and negative (e.g. correlation), never a single-hue
# sequential ramp, which makes -1 and +1 look identically "high".
DIVERGING = ["#184f95", "#6da7ec", "#f0efec", "#eb98a0", "#c23b3b"]


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": SECONDARY_INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "axes.grid": True, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11,
    })


def title_block(ax, title, subtitle=None, y=1.0):
    """Bold, finding-first title with an optional smaller context line under it.

    `title` should state the finding, not the variable being plotted --
    "Month-to-month customers churn 15x more" beats "Churn Rate by Contract".
    """
    ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK,
                 loc="left", pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5,
                 color=SECONDARY_INK, ha="left", va="bottom")


def caption(fig, text):
    """A small muted footnote at the bottom of the figure -- for 'how to read
    this chart' notes on less-familiar chart types (beeswarm, force plots)."""
    fig.text(0.01, -0.02, text, fontsize=8.5, color=MUTED, ha="left", va="top",
              wrap=True, style="italic")


# ---------------------------------------------------------------------------
# Human-readable labels for the pipeline's raw one-hot / scaled column names
# ---------------------------------------------------------------------------

_PRETTY_COLUMN = {
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly Charges ($)",
    "TotalCharges": "Total Charges ($)",
    "Contract": "Contract",
    "InternetService": "Internet Service",
    "PaymentMethod": "Payment Method",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "MultipleLines": "Multiple Lines",
    "gender": "Gender",
}

# Binary Yes/No columns (OneHotEncoder(drop="if_binary") keeps a single
# "<col>_Yes" level) read better as a short phrase than as "X: Yes".
_BINARY_PHRASE = {
    "Partner": "Has a Partner",
    "Dependents": "Has Dependents",
    "PaperlessBilling": "Paperless Billing",
    "PhoneService": "Has Phone Service",
    "SeniorCitizen": "Senior Citizen",
}


def humanize_feature_name(raw_name: str) -> str:
    """'cat__Contract_Month-to-month' -> 'Contract: Month-to-month'
    'num__tenure' -> 'Tenure (months)'
    'cat__Partner_Yes' -> 'Has a Partner'
    """
    if raw_name.startswith("num__"):
        col = raw_name[len("num__"):]
        return _PRETTY_COLUMN.get(col, col)

    if raw_name.startswith("cat__"):
        rest = raw_name[len("cat__"):]
        # Match the longest known column prefix so names containing "_" (e.g.
        # "Electronic check", "Month-to-month") split correctly.
        for col in sorted(_PRETTY_COLUMN.keys() | _BINARY_PHRASE.keys(), key=len, reverse=True):
            prefix = col + "_"
            if rest.startswith(prefix):
                category = rest[len(prefix):]
                if col in _BINARY_PHRASE and category == "Yes":
                    return _BINARY_PHRASE[col]
                pretty_col = _PRETTY_COLUMN.get(col, col)
                return f"{pretty_col}: {category}"
        return rest  # fallback: unrecognized column, show as-is

    return raw_name


def humanize_all(feature_names):
    return [humanize_feature_name(f) for f in feature_names]
