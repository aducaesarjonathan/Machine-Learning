# Historical Cocoa Price Analysis: Trends, Cycles, and Market Phases

## Overview

This project analyzes historical international cocoa prices using Python, with a focus on identifying annual price movements, investigating potential long-term price cycles, and classifying historical market phases.

The analysis uses historical cocoa price data from the International Cocoa Organization (ICCO), covering the 1969/70 to 2021/22 crop years, obtained from an Excel dataset maintained for research work at the Ghana Cocoa Board (COCOBOD).

The project applies descriptive analysis, harmonic regression, statistical hypothesis testing, and a threshold-based method for identifying bullish and bearish market phases.

## Research Objectives

The project addresses three main questions:

1. **Annual price movements:** How have international cocoa prices changed from one crop year to the next?
2. **Long-term price cycles:** Is there statistically significant evidence of a 10-year cycle in historical ICCO cocoa prices?
3. **Market phases:** Which historical periods can be classified as bullish or bearish using a 20% price-reversal threshold?

## Analyses

### 1. Year-on-Year Price Changes

The first analysis visualizes the annual percentage changes in ICCO cocoa prices using a horizontal bar chart.

**Methods**
- Load historical cocoa price data from Excel.
- Clean crop-year labels and convert percentage changes to numeric values.
- Plot positive and negative annual changes around a zero reference line.
- Annotate each bar with its corresponding percentage change.

**Output:** `cocoa_yoy_changes.png`

The visualization helps identify years of substantial price increases and declines, providing an overview of historical price volatility.

### 2. Harmonic Regression and the 10-Year Cycle Hypothesis

The second analysis investigates whether historical ICCO cocoa prices exhibit a statistically significant 10-year cycle after accounting for a linear time trend.

The model combines a linear trend with sine and cosine terms:

\[
P_t = \beta_0+\beta_1t+
\beta_2\sin\left(\frac{2\pi t}{T}\right)+
\beta_3\cos\left(\frac{2\pi t}{T}\right)+\varepsilon_t
\]

Where:

- \(P_t\) is the ICCO cocoa price in USD per tonne.
- \(t\) is the centered crop year.
- \(T\) is the hypothesized cycle length.
- \(\beta_0\) is the intercept.
- \(\beta_1\) captures the linear time trend.
- \(\beta_2\) and \(\beta_3\) capture the cyclical component.
- \(\varepsilon_t\) is the error term.

#### Statistical testing

The analysis compares two ordinary least squares (OLS) models:

- **Restricted model:** Linear time trend only.
- **Full model:** Linear trend plus harmonic terms.

A joint F-test evaluates the null hypothesis:

\[
H_0:\beta_2=\beta_3=0
\]

The alternative hypothesis is that at least one harmonic coefficient differs from zero.

The analysis first tests the 10-year cycle hypothesis at the 5% significance level. It then scans candidate cycle lengths from 4 to 25 years.

To address multiple testing, the period scan calculates a Bonferroni-adjusted significance threshold.

**Output:** `harmonic_10yr_fit.png`

The fitted model and statistical results help assess whether a 10-year cycle is supported by the historical data. A statistically significant harmonic component would indicate cyclical variation beyond the fitted linear trend, but would not by itself establish a stable economic cycle or identify its causes.

### 3. Historical Bullish and Bearish Market Phases

The third analysis identifies bullish and bearish phases using a 20% peak-to-trough or trough-to-peak reversal rule.

#### Classification methodology

- **Bullish phase:** A price increase of at least 20% from a tracked trough confirms an upward market phase.
- **Bearish phase:** A price decrease of at least 20% from a tracked peak confirms a downward market phase.

The algorithm tracks successive peaks and troughs, identifies phase transitions, and records the start year, end year, starting price, ending price, percentage change, and duration.

The analysis also calculates summary statistics, including the number of confirmed bullish and bearish phases, average phase duration, longest phase duration, and average price change.

Phases still in progress at the end of the dataset are labelled as open phases and excluded from confirmed-phase summary counts.

**Outputs:**
- `cocoa_bull_bear_phases.xlsx`
- `cocoa_bull_bear_phases.png`

The Excel file contains the identified market phases, while the chart visualizes historical price movements and the detected phase turning points.

## Dataset

**Input file:** `producer prices.xlsx`

**Worksheet:** `Sheet1`

**Expected columns:**

| Column | Description |
|---|---|
| `crop_year` | Cocoa crop-year label |
| `producer_price_cedis` | Ghana cocoa producer price recorded in the source dataset |
| `icco_price_usd` | International cocoa price in USD per tonne |
| `yoy_pct` | Year-on-year percentage change supplied in the dataset |

The scripts skip the first two worksheet rows when loading the data.

### Data preparation

The regression and market-phase scripts convert crop-year labels to numeric starting years and clean the ICCO price series.

The regression script also standardizes historical Ghana producer prices to account for the July 2007 currency redenomination, converting pre-2007/08 values from old Ghanaian cedis to Ghana cedis. Where the producer-price field contains two values separated by a slash, the script averages them.

These producer-price corrections are separate from the ICCO international price analysis.

## Technologies Used

- **Python** — Core programming language
- **Pandas** — Data loading, cleaning, and manipulation
- **NumPy** — Numerical computations and harmonic terms
- **Statsmodels** — Ordinary least squares regression
- **SciPy** — Statistical F-distribution calculations
- **Matplotlib** — Data visualization
- **OpenPyXL** — Excel output support through Pandas

## Project Structure

Organize the project as follows:

```text
cocoa-price-analysis/
├── README.md
├── producer prices.xlsx
├── yoy_analysis.py
├── harmonic_regression.py
├── bull_bear_analysis.py
├── cocoa_yoy_changes.png
├── harmonic_10yr_fit.png
├── cocoa_bull_bear_phases.png
└── cocoa_bull_bear_phases.xlsx
```

The Python filenames above are suggested names; use the actual filenames of your scripts if they differ.

## Installation and Usage

### 1. Install Python

Install Python 3.10 or a compatible version supported by your installed libraries.

### 2. Install dependencies

Run the following command in your terminal:

```bash
pip install numpy pandas statsmodels scipy matplotlib openpyxl
```

### 3. Configure the dataset path

Update `FILE_PATH` in each script to point to your local Excel file:

```python
FILE_PATH = r"D:\VS code\Cocoa Prices\producer prices.xlsx"
```

Alternatively, place the dataset in a known project directory and update the path accordingly.

### 4. Run the analyses

Execute each script independently:

```bash
python yoy_analysis.py
python harmonic_regression.py
python bull_bear_analysis.py
```

The scripts print statistical results and save the corresponding charts and Excel output to the script directory.

## Methodological Considerations

- **Descriptive rather than causal analysis:** Historical price movements and regression coefficients do not establish the economic causes of price changes.
- **Cycle interpretation:** Harmonic regression assumes a specified periodic component. A statistically significant result does not guarantee that the same cycle will persist in the future.
- **Multiple comparisons:** Searching across numerous cycle lengths increases the risk of finding an apparently significant period by chance. The Bonferroni adjustment helps control this risk.
- **Limited cycle observations:** The dataset contains approximately 53 annual observations. Consequently, long cycle lengths are represented by relatively few complete cycles, limiting the strength of conclusions about periodicity.
- **Market-phase classification:** The 20% reversal rule is a chosen analytical threshold, not a universally accepted definition of a bull or bear market. Identified turning points depend on the selected threshold and algorithm.
- **Crop-year representation:** Numeric years represent the starting year of each crop season.
- **Source data quality:** Results depend on the accuracy, consistency, and completeness of the original Excel dataset.

## Potential Extensions

Future improvements could include:

- Comparing multiple cycle lengths using out-of-sample validation.
- Examining the sensitivity of market phases to 10%, 20%, and 30% reversal thresholds.
- Testing alternative trend specifications and structural breaks.
- Investigating the relationship between cocoa prices, production, demand, inventories, and other relevant market fundamentals.
- Developing interactive dashboards for historical cocoa price monitoring.

## Disclaimer

This project is intended for historical research, statistical analysis, and educational purposes. It does not provide price forecasts, investment advice, or guaranteed evidence of recurring market cycles.

The use of COCOBOD-related data does not imply official endorsement of the analysis or its findings by the Ghana Cocoa Board.

## Author

**Jonathan Adu-Caesar**

BSc Mathematics, Kwame Nkrumah University of Science and Technology (KNUST)

Research interests: Time-series analysis, mathematical modelling, optimization, and machine learning applications in commodity markets.
