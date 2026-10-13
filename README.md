# Nike AI Sales Intelligence & Prediction — MLPA Project

An educational Streamlit dashboard for exploring Nike's reported company-level financials and demonstrating product-level analytics and machine-learning demand forecasting.

## Features
- Nike-inspired black, white, and volt dashboard styling
- Company revenue trends and channel comparisons from selected annual financial data
- Product revenue and unit ranking, date filtering, and year-over-year comparisons
- Random Forest demand forecast with chronological holdout MAE/RMSE
- Rules-based business insights and data-quality guardrails
- Upload your own product-level CSV

## Important data note
The included company financial CSV contains reported company/category/channel figures. The included product-level file `data/product_sales_demo_synthetic.csv` is **synthetic demonstration data** and does not represent actual sales of individual Nike shoe models. The app labels it accordingly. Do not describe its top/lowest models as verified real-world Nike bestsellers.

## Run on macOS
Open Terminal and change directory into this project folder, then run:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

If you already have the environment installed, you can simply run `python3 -m streamlit run app.py` after installing Plotly with `python3 -m pip install plotly`.

## Product CSV format
Required columns: `date`, `product_model`, `units_sold`, `revenue_usd`. Example date format: `2025-01-01`.

## Source
Nike FY2026 Form 10-K: https://investors.nike.com/sec-filings/all-sec-filings/content/0000320187-26-000088/nke-20260531.htm
