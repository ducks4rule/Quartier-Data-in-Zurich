import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from PIL import Image, ImageDraw

import utilities as ut


st.set_page_config(
    layout="centered",
    # layout="wide",
)
take_mean = True
# take_mean = False

key_quartier_kreis = ut.load_q_k_key()
df = ut.load_dataframe()
X_scaled, X_cols = ut.load_X_scaled()
if take_mean:
    X_df = pd.DataFrame(X_scaled)
    X_df['Quartier'] = df['Quartier'].values
    X_scaled = X_df.groupby('Quartier').mean()
labels = df['Quartier']

st.write("# PCA Analysis")
# ============================
# PCA
# ============================
pca = PCA(n_components=X_scaled.shape[0])
X_pca = pca.fit_transform(X_scaled)

# Create DataFrame for plotting
pca_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
pca_df["Quartier"] = np.unique(labels.values)

st.write('Coloring all the points by Quartier')
# Plot with Plotly
color_seq = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
fig_pca = px.scatter(
    pca_df, x="PC1", y="PC2", color="Quartier",
    title="PCA of Features Colored by Quartier",
    opacity=0.7,
    color_discrete_sequence=color_seq[:32]
)
st.plotly_chart(fig_pca, use_container_width=True)

ellbow_pos = 6
# if st.toggle("Show PCA Eigenvalues Plot"):
with st.expander("PCA Eigenvalues Plot", expanded=False):
    # plot the square of eigenvalues
    eigenvalues = pca.explained_variance_
    eigenvalues_sq = np.square(eigenvalues)
    fig_ellbow = px.line(
        x=np.arange(1, len(eigenvalues_sq) + 1),
        y=eigenvalues_sq,
        markers=True,
        labels={"x": "Principal Component", "y": "Eigenvalues<sup>2</sup>"},
        title="Eigenvalues<sup>2</sup> of PCA"
    )
    # Mark the 6th eigenvalue
    fig_ellbow.add_scatter(
        x=[ellbow_pos],
        y=[eigenvalues_sq[ellbow_pos - 1]],
        mode='markers+text',
        name='Elbow',
        marker=dict(color='red', size=10),
        text=['Elbow'],
        textposition='top right'
    )
    st.plotly_chart(fig_ellbow, use_container_width=True)

# Reduce dimension to 6 components of the pca
X_pca_trunc = X_pca[:, :ellbow_pos]
X_pca_trunc_df = pd.DataFrame(X_pca_trunc, columns=[f"PC{i+1}" for i in range(ellbow_pos)])
st.write("# Clustering the data")
st.write(f"We can reduce the dimension to {ellbow_pos} components of the PCA and use the reduced data for clustering.\n Sometimes the clusters are not intuitive, that is because we are only displaying the first two components of the PCA.")


st.write("## KMeans Clustering -- with PCA Dimension Reduction")
# Run k-means clustering on the PCA data to cluster the Quartiers
st.sidebar.write("## KMeans and Spectral Clustering")
st.sidebar.write("Select the number of clusters for the PCA data and the selected features by the tree and random forest models.")
# num_clusters = st.sidebar.radio("Number of Clusters", [2, 3, 4, 5, 6, 7, 8, 9, 10], index=3)
num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=5)

# num_clusters = st.slider("Number of Clusters for the PCA", min_value=2, max_value=10, value=5)
kmeans_pca = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_all = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_pca.fit(X_pca_trunc_df)
kmeans_all.fit(X_scaled)
col1, col2 = st.columns(2)
with col1:
    st.write(f"KMeans Clustering on PCA with {ellbow_pos} components")
    pca_df["Cluster"] = kmeans_pca.labels_
    fig_pca_clusters = px.scatter(
        pca_df, x="PC1", y="PC2", color="Cluster",
        title="PCA of Features with KMeans Clustering",
        opacity=0.7
    )
    st.plotly_chart(fig_pca_clusters, use_container_width=True)
with col2:
    st.write("KMeans Clustering on all features")
    pca_df["Cluster All"] = kmeans_all.labels_
    fig_all_clusters = px.scatter(
        pca_df, x="PC1", y="PC2", color="Cluster All",
        title="All Features with KMeans Clustering",
        opacity=0.7
    )
    st.plotly_chart(fig_all_clusters, use_container_width=True)

# difference between the two clusterings
ari = adjusted_rand_score(kmeans_pca.labels_, kmeans_all.labels_)
nmi = normalized_mutual_info_score(kmeans_pca.labels_, kmeans_all.labels_)

st.write(f'Comparing the two clusterings with the adjusted rand index and the normalized mutual information score, we get the following results:')
st.dataframe(pd.DataFrame({
    'Metric': ['Adjusted Rand Index', 'Normalized Mutual Information'],
    'Score': [ari, nmi]
}), use_container_width=False)


if st.toggle("Show definition of the metrics", value=False):
# with st.expander("Show definition of the metrics", expanded=False):
    # if st.toggle("Adjusted Rand Index (ARI)"):
    with st.expander("Adjusted Rand Index (ARI)", expanded=False):
        st.write("The adjusted Rand index (ARI) measures the similarity between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings, adjusted for chance.")
        st.latex(r"""
        \text{ARI} = \frac{ \sum_{ij} \binom{n_{ij}}{2} - \left[ \sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2} \right] / \binom{n}{2} }
        { \frac{1}{2} \left[ \sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2} \right] - \left[ \sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2} \right] / \binom{n}{2} }
        """)
        st.write("""
        Where:
        - $n_{ij}$: number of samples in both cluster $i$ of the first clustering and cluster $j$ of the second clustering  
        - $a_i = \sum_j n_{ij}$: number of samples in cluster $i$ of the first clustering  
        - $b_j = \sum_i n_{ij}$: number of samples in cluster $j$ of the second clustering  
        - $n$: total number of samples
        """)

    with st.expander("Normalized Mutual Information (NMI)", expanded=False):
        st.write("The normalized mutual information (NMI) score quantifies the agreement between the true labels and the clustering assignments, normalized to range from 0 (no mutual information) to 1 (perfect correlation).")
        st.latex(r"""
        \text{NMI}(U, V) = \frac{2 \cdot I(U; V)}{H(U) + H(V)}
        """)
        st.write("""
        Where:
        - $I(U; V)$: mutual information between clusterings $U$ and $V$  
        - $H(U)$, $H(V)$: entropies of clusterings $U$ and $V$
        """)
        st.latex(r"""
        I(U; V) = \sum_{i=1}^{|U|} \sum_{j=1}^{|V|} P(i, j) \log \left( \frac{P(i, j)}{P(i) P(j)} \right)
        """)
        st.latex(r"""
        H(U) = -\sum_{i=1}^{|U|} P(i) \log P(i)
        """)
        st.latex(r"""
        H(V) = -\sum_{j=1}^{|V|} P(j) \log P(j)
        """)
        st.write("""
        Where:
        - $P(i)$: probability a sample is in cluster $i$ of $U$  
        - $P(j)$: probability a sample is in cluster $j$ of $V$  
        - $P(i, j)$: probability a sample is in cluster $i$ of $U$ and cluster $j$ of $V$
        """)
    with st.expander("Silhouette Score", expanded=False):
        st.markdown(r"""
        The silhouette score measures how similar each data point is to its own cluster compared to other clusters, indicating the quality of clustering.

        The silhouette score for a single sample is defined as:

        $$
        s = \frac{b - a}{\max(a, b)}
        $$

        Where:
        - $ a = $ average distance from the sample to all other points in the same cluster  
        - $ b = $ lowest average distance from the sample to all points in any other cluster (i.e., the nearest cluster that the sample is not a part of)

        The overall silhouette score is the mean of $ s $ over all samples.

        See [Wikipedia](https://en.wikipedia.org/wiki/Silhouette_(clustering)) for more details.
        """)

    with st.expander("Davies-Bouldin Score", expanded=False):
        st.markdown(r"""
        The Davies-Bouldin score evaluates clustering quality by measuring the average similarity between each cluster and its most similar (i.e., least separated) cluster, with lower values indicating better clustering.

        The Davies-Bouldin index is defined as:

        $$
        DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{S_i + S_j}{M_{ij}} \right)
        $$

        Where:
        - $ k = $ number of clusters  
        - $ S_i = $ average distance of all points in cluster $ i $ to the centroid of cluster $ i $ (intra-cluster distance)  
        - $ M_{ij} = $ distance between the centroids of clusters $ i $ and $ j $ (inter-cluster distance)

        A lower Davies-Bouldin score indicates better clustering.

        See [Wikipedia](https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index) for more details.
        """)
    



st.write('## Selected Features and Spectral Clustering')
st.write("#### KMeans Clustering -- with the 'most important' features")
st.write("The 'most important' features here are the features ranked the highest a random forest model and a regression tree.")
st.page_link('pages/1_Feature_Correlation.py')
df_in_both = st.session_state['df_in_both']

# num_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=5) # Slider for number of clusters
num_features = st.sidebar.slider("Number of Features (Tree/ Forest)", min_value=1, max_value=df_in_both.shape[0], value=8)
df_features = df[df_in_both['Feature'].values].fillna(0)
df_features = df_features.iloc[:, :num_features]
df_features = StandardScaler().fit_transform(df_features)
if take_mean:
    X_df = pd.DataFrame(df_features)
    X_df['Quartier'] = df['Quartier'].values
    df_features = X_df.groupby('Quartier').mean()

kmeans_features = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_features.fit(df_features)



spectral = SpectralClustering(n_clusters=num_clusters, random_state=42, affinity='nearest_neighbors')
spectral_labels = spectral.fit_predict(X_scaled)
# spectral.fit(df_features)

col1, col2 = st.columns(2)
with col1:
    st.write(f"KMeans Clustering on the {num_features} most important features displayed on the first two principal components")
with col2:
    st.write(f"Spectral Clustering displayed on the first two principal components")

col1, col2 = st.columns(2)
with col1:
    with st.spinner("Plotting the clusters..."):
        pca_df["Cluster Features"] = kmeans_features.labels_
        fig_feat_clusters = px.scatter(
            pca_df, x="PC1", y="PC2", color="Cluster Features",
            title="Clustering on the most important features",
            opacity=0.7,
        )
        st.plotly_chart(fig_feat_clusters, use_container_width=True)

with col2:
    with st.spinner("Plotting the clusters..."):
        pca_df["Cluster Features"] = spectral.labels_
        fig_spectral_clusters = px.scatter(
            pca_df, x="PC1", y="PC2", color="Cluster Features",
            title="Spectral Clustering",
            opacity=0.7,
        )
        st.plotly_chart(fig_spectral_clusters, use_container_width=True)






# visualizing the clusters on a map of zurich
# number of distinct colors for the clusters
st.write("## Visualizing the clusters on a map of Zurich")
st.write("The clusters are visualized on a map of Zurich, the color of the circles represents the cluster that appears most often with the corresponding Quartier.")
size_circles = 30

col1, col2 = st.columns(2)
with col1:
    st.write(f"#### Clustered after PCA -- truncated at {ellbow_pos} components")
with col2:
    st.write("#### Clustered after PCA -- all features")


col1, col2= st.columns(2)
with col1:
    with st.spinner("Plotting the clusters..."):
        quartier_main_cluster_pca = (
            pca_df.groupby("Quartier")["Cluster"]
            .agg(lambda x: x.value_counts().idxmax())
        )
        colors = px.colors.qualitative.Plotly[:num_clusters]
        map_img = Image.open("Karte_Gemeinde_Zürich_Quartiere.png").convert("RGBA")
        draw = ImageDraw.Draw(map_img)

# for q in key_quartier_kreis['Quartier']:
        for q in labels.values:
            color = colors[quartier_main_cluster_pca.loc[q]]
            circle_val = size_circles
            ut.draw_in_quartier(q, draw=draw, r=circle_val, width=1, outline=color, fill=color)

        st.image(map_img)


with col2:
    with st.spinner("Plotting the clusters..."):
        quartier_main_cluster_all = (
            pca_df.groupby("Quartier")["Cluster All"]
            .agg(lambda x: x.value_counts().idxmax())
        )
        colors = px.colors.qualitative.Plotly[:num_clusters]
        map_img = Image.open("Karte_Gemeinde_Zürich_Quartiere.png").convert("RGBA")
        draw = ImageDraw.Draw(map_img)
        for q in labels.values:
            color = colors[quartier_main_cluster_all.loc[q]]
            circle_val = size_circles
            ut.draw_in_quartier(q, draw=draw, r=circle_val, width=1, outline=color, fill=color)
        st.image(map_img)


col1, col2 = st.columns(2)
with col1:
    st.write(f"#### Clustered on the {num_features} most important features ranked by arborial estimators")
with col2:
    st.write(f"#### Clustered with Spectral Clustering")

col1, col2 = st.columns(2)
with col1:
    with st.spinner("Plotting the clusters..."):
        quartier_main_cluster_features = (
            pca_df.groupby("Quartier")["Cluster Features"]
            .agg(lambda x: x.value_counts().idxmax())
        )
        colors = px.colors.qualitative.Plotly[:num_clusters]
        map_img = Image.open("Karte_Gemeinde_Zürich_Quartiere.png").convert("RGBA")
        draw = ImageDraw.Draw(map_img)
        for q in labels.values:
            color = colors[quartier_main_cluster_features.loc[q]]
            circle_val = size_circles
            ut.draw_in_quartier(q, draw=draw, r=circle_val, width=1, outline=color, fill=color)

        st.image(map_img)
with col2:
    with st.spinner("Plotting the clusters..."):
        quartier_main_cluster_spectral = (
            pca_df.groupby("Quartier")["Cluster Features"]
            .agg(lambda x: x.value_counts().idxmax())
        )
        colors = px.colors.qualitative.Plotly[:num_clusters]
        map_img = Image.open("Karte_Gemeinde_Zürich_Quartiere.png").convert("RGBA")
        draw = ImageDraw.Draw(map_img)
        for q in labels.values:
            color = colors[quartier_main_cluster_spectral.loc[q]]
            circle_val = size_circles
            ut.draw_in_quartier(q, draw=draw, r=circle_val, width=1, outline=color, fill=color)

        st.image(map_img)

col1, col2 = st.columns(2)
with col1:
    st.write(f'Comparing the three clusterings with the adjusted rand index and the normalized mutual information score, we get the following results:')
with col2:
    st.write('The unsupervised scores Silhouette score and Davies-Bouldin score give the follwoing results:')

col1, col2 = st.columns(2)
with col1:
    ari_pca_feat = adjusted_rand_score(kmeans_pca.labels_, kmeans_features.labels_)
    ari_all_feat = adjusted_rand_score(kmeans_all.labels_, kmeans_features.labels_)
    nmi_pca_feat = normalized_mutual_info_score(kmeans_pca.labels_, kmeans_features.labels_)
    nmi_all_feat = normalized_mutual_info_score(kmeans_all.labels_, kmeans_features.labels_)
    ari_pca_spec = adjusted_rand_score(kmeans_pca.labels_, spectral.labels_)
    ari_all_spec = adjusted_rand_score(kmeans_all.labels_, spectral.labels_)
    ari_feat_spec = adjusted_rand_score(kmeans_features.labels_, spectral.labels_)
    nmi_pca_spec = normalized_mutual_info_score(kmeans_pca.labels_, spectral.labels_)
    nmi_all_spec = normalized_mutual_info_score(kmeans_all.labels_, spectral.labels_)
    nmi_feat_spec = normalized_mutual_info_score(kmeans_features.labels_, spectral.labels_)
    st.dataframe(pd.DataFrame({
        'Comparison': ['PCA vs All', 'PCA vs Features', 'All vs Features', 'PCA vs Spectral', 'All vs Spectral', 'Features vs Spectral'],
        'ARI': [ari, ari_pca_feat, ari_all_feat, ari_pca_spec, ari_all_spec, ari_feat_spec],
        'NMI': [nmi, nmi_pca_feat, nmi_all_feat, nmi_pca_spec, nmi_all_spec, nmi_feat_spec]
    }), use_container_width=False)


with col2:
    sil_sc_pca = silhouette_score(X_pca_trunc_df, kmeans_pca.labels_)
    sil_sc_all = silhouette_score(X_scaled, kmeans_all.labels_)
    sil_sc_feat = silhouette_score(df_features, kmeans_features.labels_)
    sil_sc_spec = silhouette_score(X_scaled, spectral.labels_)
    db_sc_pca = davies_bouldin_score(X_pca_trunc_df, kmeans_pca.labels_)
    db_sc_all = davies_bouldin_score(X_scaled, kmeans_all.labels_)
    db_sc_feat = davies_bouldin_score(df_features, kmeans_features.labels_)
    db_sc_spec = davies_bouldin_score(X_scaled, spectral.labels_)

    st.dataframe(pd.DataFrame({
        'Clustering': ['PCA', 'All', 'Features', 'Spectral'],
        'Silhouette': [sil_sc_pca, sil_sc_all, sil_sc_feat, sil_sc_spec],
        'Davies Bouldin': [db_sc_pca, db_sc_all, db_sc_feat, db_sc_spec]
    }), use_container_width=False)

# comparing the three algorithms over the range of clusters 2 - 10
st.write("## Comparing the four different clusterings over the range of clusters 2 - 10")
df_all_res = pd.DataFrame(columns=['Metric', 'PCA vs All', 'PCA vs Features', 'All vs Features', 'PCA vs Spectral', 'All vs Spectral', 'Features vs Spectral', '# Clusters'])
df_all_res_unsupervised = pd.DataFrame(columns=['Metric', 'PCA', 'All', 'Features', 'Spectral', '# Clusters'])
cluster_df = pd.DataFrame(index=X_scaled.index)

st.write("#### Consistency between the different clusterings")

for i in range(2, 10):
    kmeans_pca = KMeans(n_clusters=i, random_state=42)
    kmeans_all = KMeans(n_clusters=i, random_state=42)
    kmeans_features = KMeans(n_clusters=i, random_state=42)
    spectral = SpectralClustering(n_clusters=i, random_state=42, affinity='nearest_neighbors')
    kmeans_pca.fit(X_pca_trunc_df)
    kmeans_all.fit(X_scaled)
    kmeans_features.fit(df_features)
    spectral.fit(X_scaled)

    cluster_df[f'kmeans_pca_{i}'] = kmeans_pca.labels_
    cluster_df[f'kmeans_all_{i}'] = kmeans_all.labels_
    cluster_df[f'kmeans_features_{i}'] = kmeans_features.labels_
    cluster_df[f'spectral_{i}'] = spectral.labels_


    ari_pca_feat = adjusted_rand_score(kmeans_pca.labels_, kmeans_features.labels_)
    ari_all_feat = adjusted_rand_score(kmeans_all.labels_, kmeans_features.labels_)
    nmi_pca_feat = normalized_mutual_info_score(kmeans_pca.labels_, kmeans_features.labels_)
    nmi_all_feat = normalized_mutual_info_score(kmeans_all.labels_, kmeans_features.labels_)
    ari_pca_spec = adjusted_rand_score(kmeans_pca.labels_, spectral.labels_)
    ari_all_spec = adjusted_rand_score(kmeans_all.labels_, spectral.labels_)
    ari_feat_spec = adjusted_rand_score(kmeans_features.labels_, spectral.labels_)
    nmi_pca_spec = normalized_mutual_info_score(kmeans_pca.labels_, spectral.labels_)
    nmi_all_spec = normalized_mutual_info_score(kmeans_all.labels_, spectral.labels_)
    nmi_feat_spec = normalized_mutual_info_score(kmeans_features.labels_, spectral.labels_)


    sil_sc_pca = silhouette_score(X_pca_trunc_df, kmeans_pca.labels_)
    sil_sc_all = silhouette_score(X_scaled, kmeans_all.labels_)
    sil_sc_feat = silhouette_score(df_features, kmeans_features.labels_)
    db_sc_pca = davies_bouldin_score(X_pca_trunc_df, kmeans_pca.labels_)
    db_sc_all = davies_bouldin_score(X_scaled, kmeans_all.labels_)
    db_sc_feat = davies_bouldin_score(df_features, kmeans_features.labels_)
    sil_sc_spec = silhouette_score(X_scaled, spectral.labels_)
    db_sc_spec = davies_bouldin_score(X_scaled, spectral.labels_)


    df_all_res = pd.concat([df_all_res, pd.DataFrame({
        'Metric': ['ARI', 'NMI'],
        'PCA vs All': [ari, nmi],
        'PCA vs Features': [ari_pca_feat, nmi_pca_feat],
        'All vs Features': [ari_all_feat, nmi_all_feat],
        'PCA vs Spectral': [ari_pca_spec, nmi_pca_spec],
        'All vs Spectral': [ari_all_spec, nmi_all_spec],
        'Features vs Spectral': [ari_feat_spec, nmi_feat_spec],
        '# Clusters': i,
    })], ignore_index=True)

    df_all_res_unsupervised = pd.concat([df_all_res_unsupervised, pd.DataFrame({
        'Metric': ['Silhouette Score', 'Davies Bouldin Score'],
        'PCA': [sil_sc_pca, db_sc_pca],
        'All': [sil_sc_all, db_sc_all],
        'Features': [sil_sc_feat, db_sc_feat],
        'Spectral': [sil_sc_spec, db_sc_spec],
        '# Clusters': i,
    })], ignore_index=True)

    # save the cluster assignments for each clustering in a dataframe
    cluster_df[f'kmeans_pca_{i}'] = kmeans_pca.labels_
    
# find the max in 'PCA vs All', 'PCA vs Features', 'All vs Features' and display a red dot there
max_ari_values = df_all_res[df_all_res['Metric'] == 'ARI'].iloc[:, 1:-1].max(axis=0)
max_nmi_values = df_all_res[df_all_res['Metric'] == 'NMI'].iloc[:, 1:-1].max(axis=0)
max_ari_clusters = []
max_nmi_clusters = []
max_silhouette_values = df_all_res_unsupervised[df_all_res_unsupervised['Metric'] == 'Silhouette Score'].iloc[:, 1:-1].max(axis=0)
max_davies_bouldin_values = df_all_res_unsupervised[df_all_res_unsupervised['Metric'] == 'Davies Bouldin Score'].iloc[:, 1:-1].min(axis=0)
max_silhouette_clusters = []
max_davies_bouldin_clusters = []
names = []
for col in df_all_res.columns[1:-1]:
    max_idx = df_all_res[df_all_res['Metric'] == 'ARI'][col].idxmax()
    max_ari_clusters.append(df_all_res.loc[max_idx, '# Clusters'])
    max_idx = df_all_res[df_all_res['Metric'] == 'NMI'][col].idxmax()
    max_nmi_clusters.append(df_all_res.loc[max_idx, '# Clusters'])
    names.append(col)
names_unsupervised = []
for col in df_all_res_unsupervised.columns[1:-1]:
    max_idx = df_all_res_unsupervised[df_all_res_unsupervised['Metric'] == 'Silhouette Score'][col].idxmax()
    max_silhouette_clusters.append(df_all_res_unsupervised.loc[max_idx, '# Clusters'])
    max_idx = df_all_res_unsupervised[df_all_res_unsupervised['Metric'] == 'Davies Bouldin Score'][col].idxmin()
    max_davies_bouldin_clusters.append(df_all_res_unsupervised.loc[max_idx, '# Clusters'])
    names_unsupervised.append(col)

    
cols = df_all_res.columns[1:-1]
fig_all_res = px.line(
    df_all_res,
    x='# Clusters',
    y=cols,
    color='Metric',
    title='Comparison of the three clusterings over the range of clusters 2 - 10'
)
# add the max points
for i in range(len(max_ari_values)):
    cluster_ari = max_ari_clusters[i]
    cluster_nmi = max_nmi_clusters[i]
    fig_all_res.add_scatter(
        x=[cluster_ari],
        y=[max_ari_values.iloc[i]],
        mode='markers+text',
        name='Max ARI - ' + names[i],
        marker=dict(color='red', size=10),
        text=['Max ARI - ' + names[i]],
        textposition='top right'
    )
    fig_all_res.add_scatter(
        x=[cluster_nmi],
        y=[max_nmi_values.iloc[i]],
        mode='markers+text',
        name='Max NMI - ' + names[i],
        marker=dict(color='green', size=10),
        text=['Max NMI - ' + names[i]],
        textposition='top right'
    )
st.plotly_chart(fig_all_res, use_container_width=True)
st.write('We can see that the clusterings on the different reductions agree with each other reasonably well (close to 1, not randomly assigned')


st.write("#### General quality of the clusterings")

cols = df_all_res_unsupervised.columns[1:-1]
fig_all_res_unsupervised = px.line(
    df_all_res_unsupervised,
    x='# Clusters',
    y=cols,
    color='Metric',
    title='Comparison of the three clusterings over the range of clusters 2 - 10'
)
# add the max points
for i in range(len(max_silhouette_values)):
    cluster_silhouette = max_silhouette_clusters[i]
    cluster_davies_bouldin = max_davies_bouldin_clusters[i]
    fig_all_res_unsupervised.add_scatter(
        x=[cluster_silhouette],
        y=[max_silhouette_values.iloc[i]],
        mode='markers+text',
        name='Max Silhouette - ' + names_unsupervised[i],
        marker=dict(color='red', size=10),
        text=['Max Silhouette - ' + names_unsupervised[i]],
        textposition='top right'
    )
    fig_all_res_unsupervised.add_scatter(
        x=[cluster_davies_bouldin],
        y=[max_davies_bouldin_values.iloc[i]],
        mode='markers+text',
        name='Min Davies Bouldin - ' + names_unsupervised[i],
        marker=dict(color='green', size=10),
        text=['Min Davies Bouldin - ' + names_unsupervised[i]],
        textposition='top right'
    )
st.plotly_chart(fig_all_res_unsupervised, use_container_width=True)

st.markdown(r''' 
The unsupervised scores are all in the good bis medium quality clustering range in both metrics.
            ''')
st.page_link("pages/3_Quartier_Comparison.py")


st.session_state['max_silhouette_values'] = max_silhouette_values
st.session_state['max_silhouette_clusters'] = max_silhouette_clusters
st.session_state['max_davies_bouldin_values'] = max_davies_bouldin_values
st.session_state['max_davies_bouldin_clusters'] = max_davies_bouldin_clusters
st.session_state['df_all_clusters'] = cluster_df


st.markdown(r'''
    #### Interpretation of the clustering metrics
    | Silhouette Score | Davies-Bouldin Score | Interpretation                                                                                  |
    |------------------|---------------------|-------------------------------------------------------------------------------------------------|
    | 0.7 – 1.0        | 0 – 0.5             | Excellent clustering: clusters are well-separated and compact.                                  |
    | 0.5 – 0.7        | 0.5 – 1.0           | Good clustering: clusters are reasonably well-separated.                                        |
    | 0.25 – 0.5       | 1.0 – 2.0           | Fair clustering: some overlap between clusters, may need improvement.                           |
    | 0.0 – 0.25       | 2.0+                | Poor clustering: clusters are not well-separated, significant overlap or ambiguity.             |
    | < 0              | High                | Incorrect clustering: samples may be assigned to the wrong clusters.                            |

            ''')
