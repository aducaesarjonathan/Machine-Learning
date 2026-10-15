"""
Harmonic Time-Series Regression: Testing the 10-Year Cocoa Price Cycle Hypothesis
COCOBOD Research Department

Data: producer_prices.xlsx (Historical Cocoa Producer Price and World Market
      Price, 1969/70 to 2021/22 crop years)

This script:
  1. Loads and cleans the raw crop-year data
  2. Builds the harmonic regression model to test a specific cycle length (T)
  3. Runs a joint F-test on the sine/cosine terms -> answers "is a T-year
     cycle statistically real?"
  4. Scans many candidate cycle lengths to see which (if any) fits best
  5. Plots the result

Install once if needed:
  pip install pandas openpyxl statsmodels matplotlib numpy --break-system-packages
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as st
import matplotlib
import matplotlib.pyplot as plt
# Note: no matplotlib.use("Agg") here -- that forces file-only saving with no
# pop-up window. Leaving the backend as default lets plt.show() below open an
# actual window when you run this on your own machine.

import os

# ---------------------------------------------------------------------------
# STEP 1: LOAD AND CLEAN THE DATA
# ---------------------------------------------------------------------------
# Put producer_prices.xlsx in the SAME FOLDER as this script, or change the
# path below to point wherever you saved it (e.g. "C:/Users/you/Documents/producer_prices.xlsx")
FILE_PATH = r"D:\VS code\Cocoa Prices\producer prices.xlsx"


raw = pd.read_excel(
    FILE_PATH,
    sheet_name="Sheet1",
    skiprows=2,
    header=None,
    names=[
        "crop_year",
        "producer_price_cedis",
        "icco_price_usd",
        "yoy_pct"
    ]
)

# Remove empty rows
raw = raw.dropna(subset=["crop_year"]).reset_index(drop=True)


# ============================================================
# 2. CLEAN THE CROP YEAR
# ============================================================

def clean_year(x):
    return str(x).lstrip("*").strip()


raw["crop_year"] = raw["crop_year"].apply(clean_year)


# ============================================================
# 3. CLEAN YoY PERCENTAGE
# ============================================================

raw["yoy_pct"] = pd.to_numeric(
    raw["yoy_pct"],
    errors="coerce"
)

# Remove rows where YoY is missing
df = raw.dropna(subset=["yoy_pct"]).copy()


# ============================================================
# 4. CREATE THE HORIZONTAL BAR CHART
# ============================================================

fig, ax = plt.subplots(figsize=(12, 18), dpi=200)

bars = ax.barh(
    df["crop_year"],
    df["yoy_pct"]
)


# ============================================================
# 5. ADD PERCENTAGE LABELS NEXT TO EACH BAR
# ============================================================

for bar, value in zip(bars, df["yoy_pct"]):

    y = bar.get_y() + bar.get_height() / 2

    if value >= 0:

        ax.text(
            value + 0.8,
            y,
            f"{value:.2f}%",
            va="center",
            ha="left",
            fontsize=8
        )

    else:

        ax.text(
            value - 0.8,
            y,
            f"{value:.2f}%",
            va="center",
            ha="right",
            fontsize=8
        )


# ============================================================
# 6. ADD ZERO LINE
# ============================================================

ax.axvline(
    x=0,
    linewidth=1
)


# ============================================================
# 7. TITLES AND LABELS
# ============================================================

ax.set_title(
    "Year-on-Year Changes in ICCO Cocoa Prices",
    fontsize=16,
    fontweight="bold",
    pad=15
)

ax.set_xlabel(
    "Percentage Change (%)",
    fontsize=11
)

ax.set_ylabel(
    "Crop Year",
    fontsize=11
)


# ============================================================
# 8. GRID
# ============================================================

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.3
)

ax.set_axisbelow(True)


# ============================================================
# 9. IMPROVE SPACING
# ============================================================

plt.tight_layout()


# ============================================================
# 10. SAVE FIGURE
# ============================================================

output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cocoa_yoy_changes.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Chart saved to:\n{output_path}")


# ============================================================
# 11. DISPLAY
# ============================================================

plt.show()

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        f"Could not find the data file at:\n  {FILE_PATH}\n"
        "Either move producer_prices.xlsx into this script's folder, "
        "or edit the FILE_PATH variable above to point to it."
    )

# skiprows=2 skips the title row and the header row (we set our own column names)
raw = pd.read_excel(
    FILE_PATH,
    sheet_name="Sheet1",
    skiprows=2,
    header=None,
    names=["crop_year", "producer_price_cedis", "icco_price_usd", "yoy_pct"],
)
raw = raw.dropna(subset=["crop_year"]).reset_index(drop=True)


def parse_start_year(crop_year_str: str) -> int:
    """'1969/1970' -> 1969, '*2007/08' -> 2007, '2000/01' -> 2000"""
    cleaned = str(crop_year_str).lstrip("*").strip()
    return int(cleaned.split("/")[0])


def parse_producer_price(value, start_year: int) -> float:
    """
    Handles two data-quality issues found in the raw file:
      (a) Two rows (2007/08, 2009/10) store TWO values as a string like
          '950.00/1,200.00' -- an in-season price revision. We average them.
      (b) Ghana redenominated its currency in July 2007 (10,000 old Cedis =
          1 new Ghana Cedi). Every value before 2007/08 must be divided by
          10,000 so the whole series sits on one consistent currency basis.
          Without this correction, the series shows a fake ~99.99% "crash"
          between 2006/07 and 2007/08 that is really just the redenomination.
    """
    if isinstance(value, str) and "/" in value:
        parts = [float(p.replace(",", "")) for p in value.split("/")]
        value = sum(parts) / len(parts)
    else:
        value = float(value)
    if start_year < 2007:
        value = value / 10000.0
    return value


raw["year"] = raw["crop_year"].apply(parse_start_year)
raw["producer_price_ghs"] = raw.apply(
    lambda row: parse_producer_price(row["producer_price_cedis"], row["year"]), axis=1
)
raw["icco_price_usd"] = raw["icco_price_usd"].astype(float)

df = raw[["year", "producer_price_ghs", "icco_price_usd"]].sort_values("year").reset_index(drop=True)
n = len(df)
print(f"Loaded {n} crop-years: {df['year'].min()} to {df['year'].max()}\n")

# ---------------------------------------------------------------------------
# STEP 2: BUILD THE HARMONIC REGRESSION MODEL FOR A SPECIFIC CYCLE LENGTH
# ---------------------------------------------------------------------------
# Model:  Price = b0 + b1*t + b2*sin(2*pi*t/T) + b3*cos(2*pi*t/T)
#   - t is time, CENTERED on its mean (keeps the regression numerically stable)
#   - b1 captures the long-run linear trend
#   - b2, b3 together capture a wave of period T, at ANY phase/starting point,
#     because b2*sin(x) + b3*cos(x) = R*sin(x + phi) for some amplitude R and
#     phase phi -- so we don't need to know in advance where the cycle "starts"

y = df["icco_price_usd"].values  # world market price -- the right series for
                                   # testing a GLOBAL price cycle (unlike the
                                   # Cedi producer price, it isn't distorted
                                   # by 50+ years of local currency inflation)
t = df["year"].values - df["year"].values.mean()


def fit_harmonic_model(t, y, T):
    """Fit trend+cycle model for cycle length T. Returns the fitted model
    plus the restricted (trend-only) model needed for the joint F-test."""
    sin_term = np.sin(2 * np.pi * t / T)
    cos_term = np.cos(2 * np.pi * t / T)

    X_restricted = sm.add_constant(t)
    X_full = sm.add_constant(np.column_stack([t, sin_term, cos_term]))

    model_restricted = sm.OLS(y, X_restricted).fit()
    model_full = sm.OLS(y, X_full).fit()
    return model_restricted, model_full


def joint_cycle_test(model_restricted, model_full, n):
    """
    Joint F-test: is the CYCLE (both sine and cosine terms together)
    statistically different from zero, beyond the linear trend alone?
    This is the correct test -- checking b2 and b3's individual p-values
    separately is not, because the true cycle's phase is unknown and could
    load onto either term.
    """
    sse_restricted = np.sum(model_restricted.resid ** 2)
    sse_full = np.sum(model_full.resid ** 2)
    q = 2                      # number of restrictions being tested (b2=0, b3=0)
    k = 4                      # parameters in the full model
    df_denom = n - k
    F = ((sse_restricted - sse_full) / q) / (sse_full / df_denom)
    p_value = 1 - st.f.cdf(F, q, df_denom)
    return F, p_value, df_denom


# ---------------------------------------------------------------------------
# STEP 3: RUN THE DIRECT TEST FOR YOUR SPECIFIC HYPOTHESIS -- T = 10 YEARS
# ---------------------------------------------------------------------------
T_HYPOTHESIS = 10
model_r, model_f = fit_harmonic_model(t, y, T_HYPOTHESIS)
F_stat, p_value, df_denom = joint_cycle_test(model_r, model_f, n)

print("=" * 70)
print(f"DIRECT TEST: Does a {T_HYPOTHESIS}-year cycle exist?")
print("=" * 70)
print(model_f.summary())
print(f"\nJoint F-test for the {T_HYPOTHESIS}-year cycle: "
      f"F(2,{df_denom}) = {F_stat:.4f},  p = {p_value:.5f}")
if p_value < 0.05:
    print(f"=> SIGNIFICANT at 5%: evidence of a real {T_HYPOTHESIS}-year cycle.")
else:
    print(f"=> NOT significant at 5%: no evidence of a real {T_HYPOTHESIS}-year cycle.")

# ---------------------------------------------------------------------------
# STEP 4: SCAN OTHER CANDIDATE CYCLE LENGTHS (EXPLORATORY)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PERIOD SCAN: testing every candidate cycle length from 4 to 25 years")
print("=" * 70)

scan_results = []
for T_candidate in range(4, 26):
    mr, mf = fit_harmonic_model(t, y, T_candidate)
    F_c, p_c, _ = joint_cycle_test(mr, mf, n)
    scan_results.append((T_candidate, mf.rsquared, F_c, p_c))
    print(f"T={T_candidate:2d} yrs:  R2={mf.rsquared:.3f}  F={F_c:.3f}  p={p_c:.4f}")

n_tests = len(scan_results)
bonferroni_threshold = 0.05 / n_tests
best = min(scan_results, key=lambda row: row[3])
print(f"\nBest-fitting period: T={best[0]} years, p={best[3]:.4f}")
print(f"Bonferroni-adjusted significance threshold for {n_tests} tests: "
      f"{bonferroni_threshold:.5f}")
print("IMPORTANT: scanning many periods and reporting the best one is a "
      "multiple-comparisons problem. Only trust a result here if it clears "
      "the Bonferroni threshold above, AND the sample contains several full "
      "cycles of that length (n_years / T should be at least ~3-5).")

# ---------------------------------------------------------------------------
# STEP 5: PLOT THE ACTUAL DATA AGAINST THE T=10 FITTED CURVE
# ---------------------------------------------------------------------------
fitted_10yr = model_f.predict(sm.add_constant(np.column_stack([
    t, np.sin(2*np.pi*t/T_HYPOTHESIS), np.cos(2*np.pi*t/T_HYPOTHESIS)
])))

fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
ax.scatter(df["year"], y, color="#5C3A21", s=30, zorder=3, label="Actual ICCO price")
ax.plot(df["year"], fitted_10yr, color="#B03A2E", linewidth=2, zorder=2,
        label=f"Fitted 10-yr harmonic model (p={p_value:.3f}, not significant)")
ax.set_title("COCOBOD Data: ICCO World Price vs. Fitted 10-Year Cycle Model",
              fontsize=11, fontweight="bold", color="#2B1B0E")
ax.set_xlabel("Crop Year (start)")
ax.set_ylabel("Price (USD/tonne)")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper left", fontsize=8, frameon=False)
plt.tight_layout()
chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "harmonic_10yr_fit.png")
plt.savefig(chart_path, facecolor="white")
print(f"\nChart saved to {chart_path}")
plt.show()  # opens an interactive window -- close it to let the script finish

"""
Historical Bullish and Bearish Cocoa Market Periods
COCOBOD Research Department

Method:
20% peak-to-trough / trough-to-peak rule

Data:
Historical ICCO cocoa price, 1969/70 to 2021/22

A bullish phase begins when price rises by at least 20% from
the most recent significant trough.

A bearish phase begins when price falls by at least 20% from
the most recent significant peak.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================================================
# STEP 1: LOAD THE DATA
# ============================================================

FILE_PATH = r"D:\VS code\Cocoa Prices\producer prices.xlsx"

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        f"Could not find the data file at:\n{FILE_PATH}"
    )


raw = pd.read_excel(
    FILE_PATH,
    sheet_name="Sheet1",
    skiprows=2,
    header=None,
    names=[
        "crop_year",
        "producer_price_cedis",
        "icco_price_usd",
        "yoy_pct"
    ]
)

raw = raw.dropna(subset=["crop_year"]).reset_index(drop=True)


# ============================================================
# STEP 2: CLEAN THE YEAR VARIABLE
# ============================================================

def parse_start_year(crop_year):
    """
    Examples:
    '1969/1970' -> 1969
    '*2007/08'  -> 2007
    '2000/01'   -> 2000
    """

    cleaned = str(crop_year).lstrip("*").strip()

    return int(cleaned.split("/")[0])


raw["year"] = raw["crop_year"].apply(parse_start_year)


# ============================================================
# STEP 3: CLEAN ICCO PRICE
# ============================================================

raw["icco_price_usd"] = pd.to_numeric(
    raw["icco_price_usd"],
    errors="coerce"
)

df = raw[
    ["year", "icco_price_usd"]
].dropna().sort_values("year").reset_index(drop=True)


print(
    f"Loaded {len(df)} crop-years: "
    f"{df['year'].min()} to {df['year'].max()}"
)


# ============================================================
# STEP 4: 20% BULL/BEAR ALGORITHM
# ============================================================

prices = df["icco_price_usd"].values
years = df["year"].values

threshold = 0.20

phases = []

# Start by assuming the market is neither bullish nor bearish.
# We wait until we observe a 20% movement.

direction = None

# Candidate peak and trough
peak_price = prices[0]
peak_year = years[0]

trough_price = prices[0]
trough_year = years[0]


for i in range(1, len(prices)):

    price = prices[i]
    year = years[i]

    # --------------------------------------------------------
    # NO MARKET DIRECTION YET
    # --------------------------------------------------------

    if direction is None:

        # New high
        if price > peak_price:
            peak_price = price
            peak_year = year

        # New low
        if price < trough_price:
            trough_price = price
            trough_year = year

        # Price has risen 20% from the trough
        if price >= trough_price * (1 + threshold):

            direction = "Bullish"

            # The bullish phase begins at the trough
            bull_start_year = trough_year
            bull_start_price = trough_price

            # Current price becomes the current peak
            peak_price = price
            peak_year = year

        # Price has fallen 20% from the peak
        elif price <= peak_price * (1 - threshold):

            direction = "Bearish"

            # The bearish phase begins at the peak
            bear_start_year = peak_year
            bear_start_price = peak_price

            # Current price becomes the current trough
            trough_price = price
            trough_year = year


    # --------------------------------------------------------
    # BULLISH PHASE
    # --------------------------------------------------------

    elif direction == "Bullish":

        # Continue updating the peak
        if price > peak_price:

            peak_price = price
            peak_year = year

        # Has price fallen 20% from the peak?
        elif price <= peak_price * (1 - threshold):

            # Bullish phase has ended
            change = (
                (peak_price - bull_start_price)
                / bull_start_price
            ) * 100

            duration = peak_year - bull_start_year

            phases.append({
                "Direction": "Bullish",
                "Start Year": bull_start_year,
                "End Year": peak_year,
                "Start Price": bull_start_price,
                "End Price": peak_price,
                "Percentage Change": change,
                "Duration (Years)": duration
            })

            # Switch to bearish
            direction = "Bearish"

            bear_start_year = peak_year
            bear_start_price = peak_price

            trough_price = price
            trough_year = year


    # --------------------------------------------------------
    # BEARISH PHASE
    # --------------------------------------------------------

    elif direction == "Bearish":

        # Continue updating the trough
        if price < trough_price:

            trough_price = price
            trough_year = year

        # Has price risen 20% from the trough?
        elif price >= trough_price * (1 + threshold):

            # Bearish phase has ended
            change = (
                (trough_price - bear_start_price)
                / bear_start_price
            ) * 100

            duration = trough_year - bear_start_year

            phases.append({
                "Direction": "Bearish",
                "Start Year": bear_start_year,
                "End Year": trough_year,
                "Start Price": bear_start_price,
                "End Price": trough_price,
                "Percentage Change": change,
                "Duration (Years)": duration
            })

            # Switch to bullish
            direction = "Bullish"

            bull_start_year = trough_year
            bull_start_price = trough_price

            peak_price = price
            peak_year = year


# ============================================================
# STEP 5: HANDLE THE FINAL OPEN PHASE
# ============================================================

# The final phase may not have experienced a 20% reversal
# before the dataset ends. We still report it.

if direction == "Bullish":

    change = (
        (peak_price - bull_start_price)
        / bull_start_price
    ) * 100

    duration = peak_year - bull_start_year

    phases.append({
        "Direction": "Bullish (Open)",
        "Start Year": bull_start_year,
        "End Year": years[-1],
        "Start Price": bull_start_price,
        "End Price": prices[-1],
        "Percentage Change": (
            (prices[-1] - bull_start_price)
            / bull_start_price
        ) * 100,
        "Duration (Years)": years[-1] - bull_start_year
    })


elif direction == "Bearish":

    phases.append({
        "Direction": "Bearish (Open)",
        "Start Year": bear_start_year,
        "End Year": years[-1],
        "Start Price": bear_start_price,
        "End Price": prices[-1],
        "Percentage Change": (
            (prices[-1] - bear_start_price)
            / bear_start_price
        ) * 100,
        "Duration (Years)": years[-1] - bear_start_year
    })


# ============================================================
# STEP 6: CREATE RESULTS TABLE
# ============================================================

results = pd.DataFrame(phases)


print("\n")
print("=" * 90)
print("HISTORICAL BULLISH AND BEARISH COCOA MARKET PHASES")
print("=" * 90)

print(
    results.to_string(
        index=False,
        formatters={
            "Start Price": "{:.2f}".format,
            "End Price": "{:.2f}".format,
            "Percentage Change": "{:+.2f}%".format
        }
    )
)


# ============================================================
# STEP 7: SUMMARY STATISTICS
# ============================================================

confirmed = results[
    ~results["Direction"].str.contains("Open")
]

bullish = confirmed[
    confirmed["Direction"] == "Bullish"
]

bearish = confirmed[
    confirmed["Direction"] == "Bearish"
]


print("\n")
print("=" * 90)
print("SUMMARY")
print("=" * 90)

print(f"Total confirmed phases: {len(confirmed)}")

print(f"Confirmed bullish phases: {len(bullish)}")
print(f"Confirmed bearish phases: {len(bearish)}")


if len(bullish) > 0:

    print(
        f"\nAverage bullish duration: "
        f"{bullish['Duration (Years)'].mean():.2f} years"
    )

    print(
        f"Longest bullish duration: "
        f"{bullish['Duration (Years)'].max():.0f} years"
    )

    print(
        f"Average bullish price change: "
        f"{bullish['Percentage Change'].mean():+.2f}%"
    )


if len(bearish) > 0:

    print(
        f"\nAverage bearish duration: "
        f"{bearish['Duration (Years)'].mean():.2f} years"
    )

    print(
        f"Longest bearish duration: "
        f"{bearish['Duration (Years)'].max():.0f} years"
    )

    print(
        f"Average bearish price change: "
        f"{bearish['Percentage Change'].mean():+.2f}%"
    )


# ============================================================
# STEP 8: SAVE THE RESULTS
# ============================================================

output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cocoa_bull_bear_phases.xlsx"
)

results.to_excel(
    output_path,
    index=False
)

print(
    f"\nResults saved to:\n{output_path}"
)


# ============================================================
# STEP 9: PLOT THE PHASES
# ============================================================

plt.figure(figsize=(12, 6), dpi=200)

plt.plot(
    years,
    prices,
    linewidth=1.8,
    label="ICCO Cocoa Price"
)

# Mark the start/end of each confirmed phase
for _, row in results.iterrows():

    if row["Direction"].startswith("Bullish"):
        marker = "^"
    else:
        marker = "v"

    plt.scatter(
        row["Start Year"],
        row["Start Price"],
        marker=marker,
        s=60,
        zorder=3
    )

    plt.scatter(
        row["End Year"],
        row["End Price"],
        marker=marker,
        s=60,
        zorder=3
    )


plt.title(
    "Historical Bullish and Bearish Cocoa Market Phases",
    fontsize=13,
    fontweight="bold"
)

plt.xlabel("Crop Year (Start Year)")
plt.ylabel("ICCO Price (USD/tonne)")

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.legend()

plt.tight_layout()

chart_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cocoa_bull_bear_phases.png"
)

plt.savefig(
    chart_path,
    facecolor="white"
)

plt.show()