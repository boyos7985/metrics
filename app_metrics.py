import streamlit as st
import pandas as pd
from metrics_functions import compute_all_metrics

st.set_page_config(layout="wide")
st.title("Equity Metrics Comparator")

col1, col2 = st.columns(2)

with col1:
    ticker1 = st.text_input("Primary ticker", "AAPL")
with col2:
    ticker2 = st.text_input("Benchmark ticker (optional)", "")

if st.button("RUN") and ticker1:

    # FIRST STOCK
    data1 = compute_all_metrics(ticker1)
    order = list(data1.keys())                     # ← we capture the order
    df1 = pd.DataFrame.from_dict(data1, orient='index', columns=[ticker1])
    df1 = df1.reindex(order)                       # ← enforce order

    if ticker2.strip() != "":
        # BENCHMARK
        data2 = compute_all_metrics(ticker2)
        df2 = pd.DataFrame.from_dict(data2, orient='index', columns=[ticker2])
        df2 = df2.reindex(order)

        df = pd.concat([df1, df2], axis=1)

        # ABS DIFF only numeric
        def diff(x,y):
            try:
                return round(float(y) - float(x),2)
            except:
                return None

        df["Abs_Diff"] = [
            diff(df.iloc[i,0], df.iloc[i,1]) for i in range(len(df))
        ]

    else:
        df = df1

    # round values 2 decimals
    df = df.applymap(lambda x: round(x,2) if isinstance(x,(int,float)) else x)

    st.dataframe(df, use_container_width=True)
