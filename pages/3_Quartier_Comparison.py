import numpy as np
import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from PIL import Image, ImageDraw

import utilities as ut

# st.set_page_config(layout="centered")
st.set_page_config(layout='wide')

max_silhouette_values = st.session_state['max_silhouette_values']
max_silhouette_clusters = st.session_state['max_silhouette_clusters']
max_davies_bouldin_values = st.session_state['max_davies_bouldin_values']
max_davies_bouldin_clusters = st.session_state['max_davies_bouldin_clusters']
df_all_cluaters = st.session_state['df_all_clusters']

# Button labels for each column
clustering_options = ["Trunc PCA", "All Feat", "Sel. Feat", "Spec. Clus."]
silhouette_buttons = max_silhouette_clusters
davies_buttons = max_davies_bouldin_clusters

if 'selected_button' not in st.session_state:
    st.session_state.selected_button = silhouette_buttons[0]
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = 0
    

# Layout
st.sidebar.markdown("### Select Cluster Count")

# Header row
header_cols = st.sidebar.columns([1, 1, 1, 1])
for i, col in enumerate(header_cols):
    col.markdown(f"<b>{clustering_options[i]}</b>", unsafe_allow_html=True)

# Silhouette row
sil_cols = st.sidebar.columns([1, 1, 1, 1])
for i, col in enumerate(sil_cols):
    if col.button(f"{silhouette_buttons[i]}", key=f"sil_{silhouette_buttons[i]}_{i}"):
        st.session_state.selected_button = silhouette_buttons[i]
        st.session_state.selected_cluster = i
st.sidebar.write("Silhouette Score Max")

# Davies-Bouldin row
db_cols = st.sidebar.columns([1, 1, 1, 1])
for i, col in enumerate(db_cols):
    # if col.button(f"{davies_buttons[i]}", key=f"db_{davies_buttons[i]}"):
    if col.button(f"{davies_buttons[i]}", key=f"db_{davies_buttons[i]}_{i}"):
        st.session_state.selected_button = davies_buttons[i]
        st.session_state.selected_cluster = i
st.sidebar.markdown("Davies Bouldin Score Min")
st.sidebar.markdown("---")

# Read out the selected value
selected_button = st.session_state.selected_button
selected_cluster = st.session_state.selected_cluster
st.sidebar.write(f"Selected: {selected_button} clusters -- {clustering_options[selected_cluster]}")











st.write('# Comparing Quartiers based on the results of the clustering')


with st.spinner('Loading selected clustering...'):
    df = ut.load_dataframe()
    ind = selected_cluster + (selected_button - 2)*len(clustering_options)

    cluster_assignments = df_all_cluaters.iloc[:, ind].reset_index()
    unknown_col = [col for col in cluster_assignments.columns if col not in ['index', 'Quartier']][0]
    number_of_clusters = int(unknown_col[-1])
    cluster_assignments = cluster_assignments.rename(columns={unknown_col: 'Cluster'})
    # num_of_cluster = st.sidebar.slider('Number of cluster to show', 0, number_of_clusters - 1, 0, 1)
    # st.dataframe(cluster_assignments[cluster_assignments['Cluster'] == num_of_cluster])

    df = pd.merge(cluster_assignments, df, left_on='Quartier', right_on='Quartier')
    df_processed = df.drop(columns=['Cluster', 'Quartier', 'Year', 'Kreis #'])
    df_sym = pd.get_dummies(df_processed.select_dtypes(include=["object", "category"]))
    df_numeric = df_processed.select_dtypes(include=[np.number])
    df_full = pd.concat([df_numeric, df_sym], axis=1)
    df_full = SimpleImputer(strategy=ut.IMPUTE_STRATEGY).fit_transform(df_full)
    scaler = StandardScaler()
    df_full = scaler.fit_transform(df_full)
    df_full = pd.DataFrame(df_full, columns=list(df_numeric.columns) + list(df_sym.columns))

    df_all_c = pd.DataFrame()
    for i in range(number_of_clusters):
        cluster_indices = df[df['Cluster'] == i].index
        cluster_frame = df_full.loc[cluster_indices]
        cluster_col = cluster_frame.columns
        cluster_frame = cluster_frame.mean()
        cluster_frame = pd.DataFrame(cluster_frame).T
        df_all_c = pd.concat([df_all_c, cluster_frame], axis=0)

    df_all_c = df_all_c.reset_index(drop=True)

    df_long = df_all_c.reset_index().melt(id_vars=['index'], var_name='Feature', value_name='Value')
    df_long['index'] = df_long['index'].astype(str)
    fig = px.bar(
        df_long,
        x='Feature',
        y='Value',
        color='index',
        barmode='group',
        color_discrete_sequence=px.colors.qualitative.Set2[:selected_button],
        title='Comparison of Quartiers based on the clustering results',
        labels={'index': 'Cluster'},
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

standout_dict = {}
standout_dict_scaled = {}
threshold = st.sidebar.slider('Threshold for features shown', 0.1, 2.0, 0.5, 0.1)

for cl in df_all_c.index:
    standout_dict[cl] = []
    standout_dict_scaled[cl] = []
    for feat in df_all_c.columns:
        this_val = df_all_c.loc[cl, feat]
        other_vals = df_all_c.loc[df_all_c.index != cl, feat]
        diff = np.mean(this_val - other_vals)
        # Unscale the diff
        feat_idx = list(df_all_c.columns).index(feat)
        diff_unscaled = diff * scaler.scale_[feat_idx]
        if abs(diff) > threshold:
            standout_dict[cl].append((feat, diff_unscaled))
            standout_dict_scaled[cl].append((feat, diff))

st.write('### Standout features for each cluster')
st.write('Compared to the average of all other clusters')

# print the standout features in a table
standout_table = ut.create_dict_to_table(standout_dict)
standout_table_scaled = ut.create_dict_to_table(standout_dict_scaled)
st.dataframe(standout_table)

# get the top 3 features for each cluster
num_top_features = st.sidebar.slider('Number of top features per cluster to display', 1, 5, 2, 1)
top_features_per_cluster = {}
for cluster in range(number_of_clusters):
    cluster_df = pd.DataFrame(standout_dict_scaled[cluster], columns=['Feature', 'Value'])
    cluster_df_unscaled = pd.DataFrame(standout_dict[cluster], columns=['Feature', 'Value'])
    top_features = (
        cluster_df.reindex(cluster_df['Value'].abs().sort_values(ascending=False).index)
        .head(num_top_features)['Feature']
        .tolist()
    )
    sorted_indices = cluster_df['Value'].abs().sort_values(ascending=False).index[:num_top_features]
    top_values = cluster_df_unscaled.loc[sorted_indices]['Value'].tolist()

    top_features_per_cluster[cluster] = top_features
    top_features_per_cluster[f'{cluster} val'] = list(np.round(top_values, 2))

col1, __, col2 = st.columns([12, 0.5, 8,])
with col1:
    circle_size = 30
    labels = cluster_assignments['Quartier']
    quartier_cluster = df.set_index("Quartier")["Cluster"]
    quartier_cluster = quartier_cluster[~quartier_cluster.index.duplicated(keep='first')]
    colors = px.colors.qualitative.Plotly[:number_of_clusters]
    map_img = Image.open("Karte_Gemeinde_Zürich_Quartiere.png").convert("RGBA")
    draw = ImageDraw.Draw(map_img)

    for q in labels.values:
        color = colors[quartier_cluster[q]]
        circle_val = circle_size
        ut.draw_in_quartier(q, draw=draw, r=circle_val, width=1, outline=color, fill=color, text=quartier_cluster[q], text_size=25)

    st.write('#### Map of Zurich with Quartiers colored by cluster')
    st.image(map_img)

with col2:
    st.write('#### The clusters are defined by the following features compared to the median of all Quartiers in the other clusters:')
    for cluster in range(number_of_clusters):
        st.write(f"Cluster {cluster}:")
        features = top_features_per_cluster[cluster]
        values = top_features_per_cluster[f"{cluster} val"]
        lines = [f"- {feature:<35}\t{value}" for i, (feature, value) in enumerate(zip(features, values))]
        st.markdown("```text\n" + "\n".join(lines) + "\n```")
        st.write("")  # Add a blank line between clusters
