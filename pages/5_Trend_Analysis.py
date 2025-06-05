import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import utilities as ut

# st.set_page_config(layout="centered")
st.set_page_config(layout='wide')


st.write('# Looking at the Features')
st.write('#### Time Series Analysis on the Features')




df = ut.load_dataframe()
# X_scaled, X_cols = ut.load_X_scaled()


# create differential features for numeric columns
df_numeric = df.drop(columns=['Kreis #']).select_dtypes(include=[np.number])
df_numeric['Quartier'] = df['Quartier']
df_sorted = df_numeric.sort_values(['Quartier', 'Year'])
df_numeric_diff = df_sorted.drop(columns=['Year']).groupby('Quartier').diff()
df_numeric_diff[['Year', 'Quartier']] = df_sorted[['Year', 'Quartier']]
df_numeric_diff = df_numeric_diff.iloc[1:]
df_numeric_diff.drop_duplicates(inplace=True)

total_cols = ['Year'] + [col for col in df_numeric.columns if 'total' in col]
df_numeric_tot = df_numeric[total_cols].drop_duplicates()
# st.write(df_numeric_tot)
col_prob = ['Year', 'Inhabitants total', 'Price Living Space total', 'Median Price total']
# df_beg = df_numeric_tot.drop(columns=col_prob).dropna(how='all')
df_numeric_tot[col_prob] = df_numeric_tot[col_prob].drop_duplicates()
# df_numeric_tot_cleaned  = pd.concat([df_prob, df_beg], axis=1)
#
# st.dataframe(df_numeric_tot_cleaned, use_container_width=True)

df_numeric_tot = df_numeric_tot.groupby('Year').sum(min_count=1)
df_numeric_tot_diff = df_numeric_diff[total_cols].drop_duplicates()
df_numeric_tot_diff = df_numeric_tot_diff.groupby('Year').sum(min_count=1)
df_numeric_tot_diff = df_numeric_tot_diff.iloc[1:]


st.session_state['df_numeric'] = df_numeric
st.session_state['df_numeric_diff'] = df_numeric_diff
st.session_state['total_cols'] = total_cols





# analysis for df_numeric_tot and df_numeric_tot_diff
std_tot = df_numeric_tot.std()
std_diff_tot = df_numeric_tot_diff.std()
std_tot_sort = std_tot.sort_values(ascending=False)
std_diff_tot_sort = std_diff_tot.sort_values(ascending=False)


# TODO: add explanation for what we will do
# with st.expander(f'Variance of the different features over the years {df["Year"].min()} - {df["Year"].max()}'):
col1, col2 = st.columns(2)
with col1:
    with st.spinner('Loading...'):
        df_plot = pd.DataFrame({
            'Feature': list(std_tot_sort.index) + list(std_diff_tot_sort.index),
            'Standard Deviation': list(std_tot_sort.values) + list(std_diff_tot_sort.values),
            'Type': ['Total Standard Deviation'] * len(std_tot_sort) + ['Diff Variance'] * len(std_diff_tot_sort)
        })

        fig = px.bar(
            df_plot,
            x='Feature',
            y='Standard Deviation',
            color='Type',
            barmode='group',
            title='Standard Deviation of the Features sorted',
            log_y=True
        )
        st.plotly_chart(fig, use_container_width=True)


# fit a linear trend to each fieature

with col2:
    with st.spinner('Loading...'):
        trend_slope_tot = ut.compute_trend_slopes(ut.scale_columns_ignore_nan(df_numeric_tot)).sort_values(by='Trend', key=lambda x: x.abs(), ascending=False)
        trend_slope_tot_diff = ut.compute_trend_slopes(ut.scale_columns_ignore_nan(df_numeric_tot_diff)).sort_values(by='Trend', key=lambda x: x.abs(), ascending=False)

        df_plot_trend = pd.DataFrame({
                'Feature': list(trend_slope_tot.index) + list(trend_slope_tot_diff.index),
                'Trend': list(trend_slope_tot.values) + list(trend_slope_tot_diff.values),
                'Type': ['Total Trend'] * len(trend_slope_tot) + ['Diff Trend'] * len(trend_slope_tot_diff)
        })
        df_plot_trend['Trend'] = df_plot_trend['Trend'].apply(lambda x: x.item() if isinstance(x, np.ndarray) and x.size == 1 else x)

        fig_trend = px.bar(
            df_plot_trend,
            x='Feature',
            y='Trend',
            color='Type',
            barmode='group',
            title='Slopes of the trends of the Features sorted',
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        # st.write('The trends also agree with the Standard Deviation, after the first 9 features, the trend is neglegible.')


# num_features = st.sidebar.number_input('Number of features for further analysis', min_value=1, max_value=len(std_tot_sort), value=9, step=1)

# st.write('After the first 9 features, the Standard Deviation drops significantly, we will focus on the first 9 features.')

st.markdown('''
Compute further features that describe the time series of the features.
- The slope and standard deviation of the differential features
- The mean of the features
- The slope, standard deviation and the maximum change of the rolling mean of the features
- The maximum and minimum of the autocorrelation of the features
''')


df_compare_tot = ut.compute_features_for_features(
    df_numeric_tot,
    df_numeric_tot_diff,
    )

n_clusters = st.sidebar.slider('Number of clusters', 1, 5, 3, 1)
with st.spinner('Fitting Dirichlet Process ...'):
    X = df_compare_tot.drop(columns=['Feature']).fillna(0).values
    X = ut.scale_columns_ignore_nan(pd.DataFrame(X)).values
    # cluster the data with spectral clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    df_compare_tot['Cluster'] = kmeans.labels_
    # spectral_cl = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', assign_labels='discretize')
    # spectral_cl.fit(X)
    # df_compare_tot['Cluster'] = spectral_cl.labels_
    df_compare_tot['Cluster'] = df_compare_tot['Cluster'].astype(str)

    # create dictionary of clusters and features
    cluster_dict = {}
    for cl in df_compare_tot['Cluster'].unique():
        cluster_dict[cl] = []
        cluster_df = df_compare_tot[df_compare_tot['Cluster'] == cl]
        for feat in cluster_df['Feature']:
            cluster_dict[cl].append((feat, 0))

    # create a dataframe with the cluster and the Features
    df_cluster = ut.create_dict_to_table(cluster_dict)
    df_cluster = df_cluster.drop(columns=[col for col in df_cluster.columns if 'Value' in col])

    scaled_array = StandardScaler().fit_transform(df_compare_tot.drop(columns=['Feature', 'Cluster']))
    df_compare_tot_scaled = pd.DataFrame(
        scaled_array,
        columns=df_compare_tot.drop(columns=['Feature', 'Cluster']).columns,
        index=df_compare_tot.index
    )
    df_compare_tot_scaled['Feature'] = df_compare_tot['Feature']
    df_compare_tot_scaled['Cluster'] = df_compare_tot['Cluster']

    cluster_means_scaled = (
        df_compare_tot_scaled
        .drop(columns=['Feature'])
        .groupby('Cluster')
        .mean()
    )
    cluster_means = (
        df_compare_tot
        .drop(columns=['Feature'])
        .groupby('Cluster')
        .mean()
    )

    df_plot_2 = cluster_means_scaled.reset_index().melt(id_vars=['Cluster'], var_name='Feature', value_name='Value')

    fig = px.bar(
        # cluster_means.reset_index().melt(id_vars=['Cluster'], var_name='Feature', value_name='Value'),
        df_plot_2,
        x='Feature',
        y='Value',
        color='Cluster',
        barmode='group',
        color_discrete_sequence=px.colors.qualitative.Set2[:n_clusters],
        title='Comparison of the Features based on the extracted time series features',
        labels={'index': 'Cluster'},
        height=600,
    )
    cols = st.columns([1, 3, 1])
    with cols[1]:
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df_cluster)

    with st.expander('Cluster Means'):
        st.dataframe(cluster_means)

# plot one cluster (all the features in one cluster)
# cluster = st.sidebar.selectbox('Select Cluster', df_compare_tot['Cluster'].unique())






fig = go.Figure()
scaler = StandardScaler()

for cl in range(n_clusters):
    cols = [col for col in df_numeric_tot.columns if col in np.array(cluster_dict[str(cl)])[:,0]]
    df_cluster_plot_sc = pd.DataFrame(
        scaler.fit_transform(df_numeric_tot[cols]),
        columns=cols,
        index=df_numeric_tot.index
    )
    df_cluster_plot = df_numeric_tot[cols]
    for col in cols:
        fig.add_trace(go.Scatter(
            x=df_cluster_plot_sc.index,
            y=df_cluster_plot_sc[col],
            mode='lines',
            name=f'Cluster {cl} - {col}',
            # visible=(cl == cluster)  # Only the selected cluster is visible initially
        ))

# Create buttons for each cluster
buttons = []
for cl in range(n_clusters):
    visibility = []
    for i in range(n_clusters):
        n_cols = len([col for col in df_numeric_tot.columns if col in np.array(cluster_dict[str(i)])[:,0]])
        visibility.extend([i == cl] * n_cols)
    buttons.append(dict(
        label=f'Cluster {cl}',
        method='update',
        args=[{'visible': visibility},
              {'title': f'Cluster {cl} Features'}]
    ))

fig.update_layout(
    updatemenus=[dict(
        type='dropdown',
        showactive=True,
        buttons=buttons,
        x=1.15,
        y=1.15
    )],
    title=f'Cluster Features',
    xaxis_title='Index',
    yaxis_title='Value'
)

cols = st.columns([1, 8, 1])
with cols[1]:
    st.plotly_chart(fig, use_container_width=True)
    # st.write(f"Columns with only NaNs: {num_all_nan_cols} / {num_total_cols}")

