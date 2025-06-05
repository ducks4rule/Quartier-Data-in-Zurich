import numpy as np
import streamlit as st
import pickle
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw

import utilities as ut


st.set_page_config(layout='wide')
# st.set_page_config(layout="centered")

st.write('# Looking at the Features')
st.write('### Quartier by Quartier')
st.write('#### Time Series Analysis on the Features')

window_width = 8
n_entries = 8

def compute_compare_q_dict(n_entries=8, remove_short=True, remove_unassigned=True):
    df_numeric = st.session_state['df_numeric']
    df_numeric_diff = st.session_state['df_numeric_diff']
    total_cols = st.session_state['total_cols']

    compare_q_dict = {}
    quartiers = df_numeric['Quartier'].unique()
    if remove_unassigned:
        quartiers_to_remove = ['Ganze Stadt', 'Unbekannt (Stadt Zürich)', 'Nicht zuordenbar']
        quartiers = [q for q in quartiers if q not in quartiers_to_remove and 'Kreis' not in q]
    num_quartiers = len(quartiers)

    for i, q in enumerate(quartiers):
        df_numeric_q = df_numeric[df_numeric['Quartier'] == q].drop(columns=['Quartier'] + total_cols).drop_duplicates()
        df_numeric_q_diff = df_numeric_diff[df_numeric_diff['Quartier'] == q].drop(columns=['Quartier'] + total_cols).drop_duplicates()
        if remove_short:
            # test_df = df_numeric_q['Year', feat]
            valid_features = df_numeric_q.columns[df_numeric_q.notna().sum() >= n_entries]
            for f in valid_features:
                if not df_numeric_q[f].drop_duplicates().notna().sum() >= n_entries:
                    valid_features = valid_features[valid_features != f]
                    
            df_numeric_q = df_numeric_q[valid_features]
            df_numeric_q_diff = df_numeric_q_diff[valid_features]
            
        trend_slope = ut.compute_trend_slopes(
            ut.scale_columns_ignore_nan(df_numeric_q)
            # df_numeric_q
        ).sort_values(by='Trend', key=lambda x: x.abs(), ascending=False)
        trend_slope_diff = ut.compute_trend_slopes(
            ut.scale_columns_ignore_nan(df_numeric_q_diff)
        ).sort_values(by='Trend', key=lambda x: x.abs(), ascending=False)
        df_rolling = df_numeric_q.rolling(window_width).mean()
        df_rolling_slope = ut.compute_trend_slopes(
            ut.scale_columns_ignore_nan(df_rolling)
        ).sort_values(by='Trend', key=lambda x: x.abs(), ascending=False)
        compare_q_dict[q] = {
            "trend_slope": trend_slope,
            "trend_slope_diff": trend_slope_diff,
            "df_rolling_slope": df_rolling_slope
        }
    return compare_q_dict

pickle_path = "data/compare_q_dict.pkl"

st.sidebar.header("Data Loading Options")
load_option = st.sidebar.radio(
    "Choose data source:",
    ("Load from file", "Recompute and save")
)

if load_option == "Load from file" and os.path.exists(pickle_path):
    with open(pickle_path, "rb") as f:
        compare_q_dict = pickle.load(f)
    st.sidebar.success("Loaded precomputed data from file.")
elif load_option == "Recompute and save":
    with st.status("Recomputing data...", expanded=True) as status:
        compare_q_dict = compute_compare_q_dict(n_entries=n_entries)
        with open(pickle_path, "wb") as f:
            pickle.dump(compare_q_dict, f)
        status.update(label="Done!", state="complete")
    st.sidebar.success("Recomputed and saved data.")
else:
    st.sidebar.error("No precomputed file found. Please recompute and save first.")



tabs = st.tabs(["Analysis of the Features", "Quartiers with biggest changes", "Most dynamic Quartiers"])

with tabs[0]:
    st.markdown(f'''
        #### Analysis of the Features
        We compute the trends, the trends of the discrete derivative and the trends of the rolling average for each feature in each Quartier.

        For each Feature compte the z-score $ z = \\frac{{x - \\mu}}{{\\sigma}} $ over the trends of the Quartiers.

        Trend interpretation:
        - $|$trend$| < 0.01$ → extremely flat
        - $0.01 ≤ |$trend$| < 0.1$ → very small trend
        - $0.1 ≤ |$trend$| < 0.5$ → moderate trend
        - $|$trend$| ≥ 0.5$ → strong trend

        Then filter the features where one Quartier has a z-score $ > \\epsilon $ and where the max slope is $ > \\delta $. We also only accept trends of time series with at least {n_entries} entries.
    ''')

    def get_filtered_trend_df(compare_q_dict, trend_type, epsilon=2.0, delta=0.02):

        trend_df = pd.DataFrame({
            quartier: data[trend_type]["Trend"].astype(float)
            for quartier, data in compare_q_dict.items()
        }).T  # Transpose so quartiers are rows

        feature_stats = trend_df.describe().T  # mean, std, min, max, etc.
        z_scores = (trend_df - trend_df.mean()) / trend_df.std()

        # Quartiers with |z-scores| > epsilon
        standouts = {}
        for feature in trend_df.columns:
            outlier_quartiers = z_scores.index[abs(z_scores[feature]) > epsilon].tolist()
            if outlier_quartiers:
                standouts[feature] = outlier_quartiers

        standout_features = list(standouts.keys())
        max_trend = trend_df[standout_features].abs().max()
        filtered_features = max_trend[max_trend > delta].index.tolist()
        max_abs_z = z_scores[filtered_features].abs().max()
        sorted_features = max_abs_z.sort_values(ascending=False).index.tolist()
        trend_df_filtered = trend_df[sorted_features]

        return trend_df_filtered

    epsilon = st.sidebar.slider('Z-score threshold', min_value=0.0, max_value=7.0, value=3.0, step=0.1)
    delta = st.sidebar.slider('Max trend is at leat', min_value=0.0, max_value=0.5, value=0.01, step=0.01)
    trend_type = st.sidebar.selectbox('Select trend type to compare Quartiers visually:', ['trend_slope', 'trend_slope_diff', 'df_rolling_slope'])

    trend_df_filtered = get_filtered_trend_df(compare_q_dict, trend_type, epsilon, delta)
    z_scores = (trend_df_filtered - trend_df_filtered.mean()) / trend_df_filtered.std()

    with st.expander(f'Plots of the Features per Quartier with z-scores > {epsilon} and max trend > {delta}', expanded=True):
        cols = st.columns(3)
        for i, feature in enumerate(trend_df_filtered.columns):
            with cols[i % 3]:
                fig = px.bar(
                    trend_df_filtered.reset_index(),
                    x='index',
                    y=feature,
                    title=f"Trend for {feature} across Quartiers",
                    labels={'index': 'Quartier', feature: 'Trend'}
                )
                st.plotly_chart(fig)
                st.write(f"Z-score for {feature}: {z_scores[feature].abs().max():.2f}")




with tabs[1]:
    st.write('### Quartiers with biggest slopes in the trend analysis')
    st.write('Displayed is the Quartier with the highest z-score in the feature and the Quartier which slope is furthest away from the slope of the former Quartier')
    col, _ = st.columns([1, 5])
    with col:
        trend_type = st.selectbox('Select trend type to compare the Quartiers visually:', ['trend_slope', 'df_rolling_slope'])
    # trend_type = 'trend_slope'
    # trrend_type = 'df_rolling_slope'
    trends_filtered = get_filtered_trend_df(compare_q_dict, trend_type, epsilon, delta)

    top_quartiers_per_feature = {}
    z_scores = (trends_filtered - trends_filtered.mean()) / trends_filtered.std()
    # for feature in trend_df_filtered.columns:
    for feature in z_scores.columns:
        z = z_scores[feature].abs()
        top_quartiers = z.sort_values(ascending=False).head(1).index.tolist()
        top_quartiers_per_feature[feature] = top_quartiers

    top_slopes_feature = {}
    for feature in trends_filtered.columns:
        slopes = trends_filtered[feature]
        slopes = pd.Series(slopes, index=trends_filtered.index).dropna()
        sorted_q = slopes.sort_values(ascending=False)
        top_quartiers_per_feature[feature] = [sorted_q.index[0], sorted_q.index[-1]]
        top_slopes_feature[feature] = [sorted_q.iloc[0], sorted_q.iloc[-1]]

    # plot the top Quartiers for each feature with plotly
    df_numeric = st.session_state['df_numeric']

    with st.spinner('Plotting...'):
        cols = st.columns(2)
        i = 0
        for feature, quartiers in top_quartiers_per_feature.items():
            with cols[i % 2]:
                i += 1
                df_plot = df_numeric[df_numeric['Quartier'].isin(quartiers)][['Year', feature, 'Quartier']].dropna().drop_duplicates()
                df_plot = df_plot.groupby(['Year', 'Quartier']).sum(min_count=1).reset_index()

                fig = go.Figure()

                for quartier in quartiers:
                    df_q = df_plot[df_plot['Quartier'] == quartier]
                    fig.add_trace(go.Scatter(
                        x=df_q['Year'],
                        y=df_q[feature],
                        mode='lines+markers',
                        name=quartier
                    ))
                    # Fit linear trend if enough points
                    if len(df_q['Year']) > 1 and df_q[feature].notna().sum() > 1:
                        x = np.array(df_q['Year'])
                        y = np.array(df_q[feature])
                        mask = ~np.isnan(x) & ~np.isnan(y)
                        x = x[mask]
                        y = y[mask]
                        if len(x) > 1:
                            coef = np.polyfit(x, y, 1)
                            trend = coef[0] * x + coef[1]
                            fig.add_trace(go.Scatter(
                                x=x,
                                y=trend,
                                mode='lines',
                                name=f"{quartier} Trend",
                                line=dict(dash='dash')
                            ))

                fig.update_layout(
                    title=f"Trend for {feature} for the Quartiers {', '.join(quartiers)}",
                    xaxis_title='Year',
                    yaxis_title='Trend'
                )
                st.plotly_chart(fig)








    key_quartier_kreis = ut.load_q_k_key()

    with st.spinner('Plotting magnitude on map...'):
        cols = st.columns(2)
        i = 0
        for feature, quartiers in top_quartiers_per_feature.items():
            with cols[i % 2]:
                i += 1
                _, col2, _ = st.columns([1, 7, 1])
                with col2:
                    years = df_numeric['Year'][df_numeric[feature].notna()].unique()
                    year = st.slider('Select Year', min_value=years.min(), max_value=years.max(), value=years.max(), step=1, key=f"year_slider_{feature}")
                    df_value = df_numeric[df_numeric['Year'] == year][[feature, 'Quartier']].drop_duplicates()
                    df_value[feature] = df_value[feature].fillna(0)
                    df_value = df_value.groupby(['Quartier']).mean().reset_index()
                    df_value[feature] = (df_value[feature] - df_value[feature].min()) / df_value[feature].std()
                    st.write(f"#### Magnitude of '{feature}' in {year} for all Quartiers")
                    st.write(f"The Quartiers {' and '.join(quartiers)} are highlighted in blue")
                    map_img = Image.open("Karte_Gemeinde_Zürich_Quartiere.png").convert("RGBA")
                    draw = ImageDraw.Draw(map_img)
                    for q in df_value['Quartier']:
                        circle_val = df_value[df_value['Quartier'] == q][feature].values
                        circle_val = 2 + circle_val * 20
                        color = 'orange'
                        ut.draw_in_quartier(q, draw=draw, r=circle_val, width=1, outline="red", fill=color)
                    for q in quartiers:
                        circle_val = df_value[df_value['Quartier'] == q][feature].values
                        circle_val = 5 + circle_val * 10
                        color = 'blue'
                        ut.draw_in_quartier(q, draw=draw, r=circle_val, width=1, outline="red", fill=color)
                    st.image(map_img)













with tabs[2]:
    st.write('### Quartiers with biggest changes in the trend analysis')
    st.write('The follwoing Quartiers have the highes z-scores in the feature in the trend analysis of the discrete derivative')

    trend_type = 'trend_slope_diff'
    trends_filtered = get_filtered_trend_df(compare_q_dict, trend_type, epsilon, delta)

    top_quartiers_per_feature = {}
    z_scores = (trends_filtered - trends_filtered.mean()) / trends_filtered.std()
    # for feature in trend_df_filtered.columns:
    for feature in z_scores.columns:
        z = z_scores[feature].abs()
        top_quartiers = z.sort_values(ascending=False).head(1).index.tolist()
        top_quartiers_per_feature[feature] = top_quartiers

    # top_quartiers_per_feature = {}
    for feature in trends_filtered.columns:
        slopes = trends_filtered[feature]
        slopes = pd.Series(slopes, index=trends_filtered.index).dropna()
        sorted_q = slopes.sort_values(ascending=False)
        top_quartiers_per_feature[feature] = [sorted_q.index[0], sorted_q.index[-1]]

    # plot the top Quartiers for each feature with plotly
    df_numeric = st.session_state['df_numeric']

    with st.spinner('Plotting...'):
        cols = st.columns(2)
        i = 0
        for feature, quartiers in top_quartiers_per_feature.items():
            with cols[i % 2]:
                i += 1
                df_plot = df_numeric[df_numeric['Quartier'].isin(quartiers)][['Year', feature, 'Quartier']].dropna().drop_duplicates()
                df_plot = df_plot.groupby(['Year', 'Quartier']).sum(min_count=1).reset_index()

                fig = go.Figure()

                for quartier in quartiers:
                    df_q = df_plot[df_plot['Quartier'] == quartier]
                    fig.add_trace(go.Scatter(
                        x=df_q['Year'],
                        y=df_q[feature],
                        mode='lines+markers',
                        name=quartier
                    ))
                    # Fit linear trend if enough points
                    if len(df_q['Year']) > 1 and df_q[feature].notna().sum() > 1:
                        x = np.array(df_q['Year'])
                        y = np.array(df_q[feature])
                        mask = ~np.isnan(x) & ~np.isnan(y)
                        x = x[mask]
                        y = y[mask]
                        if len(x) > 1:
                            coef = np.polyfit(x, y, 1)
                            trend = coef[0] * x + coef[1]
                            fig.add_trace(go.Scatter(
                                x=x,
                                y=trend,
                                mode='lines',
                                name=f"{quartier} ∂/∂t Trend {coef[0]:.2f}",
                                line=dict(dash='dash')
                            ))

                fig.update_layout(
                    title=f"Trend for the discrete derivative of {feature} for the Quartiers {', '.join(quartiers)}",
                    xaxis_title='Year',
                    yaxis_title='∂/∂t Trend'
                )
                st.plotly_chart(fig)


    with st.spinner('Plotting magnitude on map...'):
        cols = st.columns(2)
        i = 0
        for feature, quartiers in top_quartiers_per_feature.items():
            with cols[i % 2]:
                i += 1
                _, col2, _ = st.columns([1, 7, 1])
                with col2:
                    years = df_numeric['Year'][df_numeric[feature].notna()].unique()
                    year = st.slider('Select Year', min_value=years.min(), max_value=years.max(), value=years.max(), step=1, key=f"year_slider_{feature}_diff")
                    df_value = df_numeric[df_numeric['Year'] == year][[feature, 'Quartier']].drop_duplicates()
                    df_value[feature] = df_value[feature].fillna(0)
                    df_value = df_value.groupby(['Quartier']).mean().reset_index()
                    df_value[feature] = (df_value[feature] - df_value[feature].min()) / df_value[feature].std()
                    st.write(f"#### Magnitude of '{feature}' in {year} for all Quartiers")
                    st.write(f"The Quartiers {' and '.join(quartiers)} are highlighted in blue")
                    map_img = Image.open("Karte_Gemeinde_Zürich_Quartiere.png").convert("RGBA")
                    draw = ImageDraw.Draw(map_img)
                    for q in df_value['Quartier']:
                        circle_val = df_value[df_value['Quartier'] == q][feature].values
                        circle_val = 2 + circle_val * 20
                        color = 'orange'
                        ut.draw_in_quartier(q, draw=draw, r=circle_val, width=1, outline="red", fill=color)
                    for q in quartiers:
                        circle_val = df_value[df_value['Quartier'] == q][feature].values
                        circle_val = 5 + circle_val * 10
                        color = 'blue'
                        ut.draw_in_quartier(q, draw=draw, r=circle_val, width=1, outline="red", fill=color)
                    st.image(map_img)
