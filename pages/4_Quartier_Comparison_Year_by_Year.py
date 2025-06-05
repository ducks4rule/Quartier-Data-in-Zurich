from sklearn.impute import SimpleImputer
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import Isomap
from sklearn.cluster import SpectralClustering
from PIL import Image, ImageDraw

import utilities as ut

# st.set_page_config(layout="centered")
st.set_page_config(layout='wide')
st.write('# Comparing Quartiers year by year')

df = ut.load_dataframe()
X_scaled_all, X_cols = ut.load_X_scaled()
labels = df['Quartier']

years = np.arange(1993, 2024, 1)

# year_ind = -1
# year = years[year_ind]
# st.sidebar.write('Select a year to compare Quartiers:')
# year = st.sidebar.slider('Year', min_value=1993, max_value=2024, value=2024, step=1)
# remove_kreis = True
n_clusters = st.sidebar.slider('Number of clusters', min_value=2, max_value=10, value=5, step=1)

num_cols  = 3
metric = st.sidebar.radio("Select metric for Isomap", ('cosine', 'euclidean'), index=0)
default_years = [1993, 2008, 2024]
cols = st.columns(3)
for i in range(num_cols):
    with cols[i]:
        # year = st.selectbox(f"Select year", years, index=years.tolist().index(default_years[i]), key=f"year_{i}", on_change=None)
        year = st.number_input(f"Select year", min_value=1993, max_value=2024, value=default_years[i], step=1, key=f"year_{i}", on_change=None)
        df_year = ut.merge_dfs_per_year([df], years=[year])
        df_year = df_year.drop(columns=['Year', 'Kreis #'])
        labels = df['Quartier']
        X_scaled = X_scaled_all[df_year.index]

        X_df = pd.DataFrame(X_scaled)
        X_df['Quartier'] = labels.iloc[df_year.index].values
        X_grouped = X_df.groupby('Quartier').mean()

        X_grouped = X_grouped[~X_grouped.index.to_series().str.contains('Kreis')]
        X_grouped = X_grouped[~X_grouped.index.isin(['Ganze Stadt', 'Unbekannt (Stadt Zürich)', 'Nicht zuordenbar'])]

        X_iso = Isomap(n_components=2, n_neighbors=5, metric=metric).fit_transform(X_grouped.values)
        # fig = px.scatter(
        #     x=X_iso[:, 0],
        #     y=X_iso[:, 1],
        #     text=X_grouped.index,
        #     title="Isomap of Quartiers, with cosine distance",
        #     labels=dict(x="Isomap 1", y="Isomap 2"),
        # )
        # fig.update_traces(textposition='top center')
        # st.plotly_chart(fig, use_container_width=True)


# st.write('This approach yields structural insights')
# affinity_matrix = cosine_similarity(X_grouped.values)
        clustering = SpectralClustering(n_clusters=n_clusters,
                                        # affinity='nearest_neighbors',
                                        affinity='cosine',
                                        random_state=42)
        labels = clustering.fit_predict(X_grouped.values)
        X_grouped['Cluster'] = labels

        fig = px.scatter(
            x=X_iso[:, 0],
            y=X_iso[:, 1],
            # x=X_iso_euc[:, 0],
            # y=X_iso_euc[:, 1],
            text=X_grouped.index,
            color=X_grouped['Cluster'].astype(str),  # Make clusters categorical
            color_discrete_sequence=px.colors.qualitative.Set1,
            title="Isomap of Quartiers,<br> with euclidean distance and clustering",
            labels=dict(x="Isomap 1", y="Isomap 2"),
        )
        fig.update_traces(textposition='top center', marker=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)

        _, map_col, _ = st.columns([1, 5, 1])
        with map_col:
            st.write('**Clusters on the map**')
            size_circles = 30
            qs_cluster = X_grouped['Cluster']
            colors = px.colors.qualitative.Plotly[:n_clusters]
            map_img = Image.open("Karte_Gemeinde_Zürich_Quartiere.png").convert("RGBA")
            draw = ImageDraw.Draw(map_img)

# for q in key_quartier_kreis['Quartier']:
            for q in X_grouped.index:
                color = colors[qs_cluster[q]]
                circle_val = size_circles
                ut.draw_in_quartier(q, draw=draw, r=circle_val, width=1, outline=color, fill=color)

            st.image(map_img)


        num_total_cols = df_year.drop(columns=['Quartier']).shape[1]
        num_not_all_nan_cols = (~df_year.drop(columns=['Quartier']).isna().all()).sum()
        # st.write(f"Columns without only NaNs: {num_not_all_nan_cols} / {num_total_cols}")

        st.write('')
        st.markdown('---')
        if i == 0:
            st.write('#### Data availability and clustering quality')
        else:
            st.write('#### ')
# cols = st.columns(num_cols)
# for i in range(num_cols):
#     with cols[i]:
        st.metric("Features with values", f"{num_not_all_nan_cols} / {num_total_cols}")
        st.progress(num_not_all_nan_cols / num_total_cols)
        st.write(f"The other features do not contain data for the selected year {year}.")

        st.write('')
        st.write('#### Quality of the clustering')
        # calculate silhouette_score and davies_bouldin_score
        score = silhouette_score(X_grouped.values, labels)
        st.write(f"Silhouette Score: {score:.3f}")
        score = davies_bouldin_score(X_grouped.values, labels)
        st.write(f"Davies-Bouldin Index: {score:.3f}")
