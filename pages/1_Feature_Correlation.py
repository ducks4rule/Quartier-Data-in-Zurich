import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

import utilities as ut
st.set_page_config(layout="wide")


df = ut.load_dataframe()
X_scaled, X_cols = ut.load_X_scaled()



# covariance matrix
cov_mat = np.cov(X_scaled.T)

# cov_mat = df.drop(columns=["Quartier", 'Year', 'Kreis #']).corr()
st.write('## Covariance Matrix')
#plot covariance Matrix
fig = px.imshow(cov_mat, x=X_cols, y=X_cols, color_continuous_scale='RdBu', title="Covariance Matrix")
st.plotly_chart(fig, use_container_width=False)

cov_cutoff = 0.7
st.write(f'### Highly correlated features with $|$cor$(a, b)| > $ {cov_cutoff}')
# Get upper triangle of the correlation matrix, excluding the diagonal
corr_pairs = np.column_stack(np.where(np.triu(cov_mat, k=1)))
corr_values = cov_mat[corr_pairs[:, 0], corr_pairs[:, 1]]
df_corr_pairs = pd.DataFrame({
    'Feature 1': X_cols[corr_pairs[:, 0]],
    'Feature 2': X_cols[corr_pairs[:, 1]],
    'Correlation': corr_values
})

high_corr_pairs = df_corr_pairs[abs(df_corr_pairs['Correlation']) > cov_cutoff]
high_corr_pairs = high_corr_pairs.reindex(high_corr_pairs['Correlation'].abs().sort_values(ascending=False).index)

feature_counts = Counter(high_corr_pairs['Feature 1']).copy()
feature_counts.update(high_corr_pairs['Feature 2'])
most_common_features = feature_counts.most_common()

# features with low correlation with all other features
cov_mat_no_diag = cov_mat.copy()
np.fill_diagonal(cov_mat_no_diag, 0)
row_max = np.max(np.abs(cov_mat_no_diag), axis=1)
low_corr_features = np.where(row_max < 0.5)[0]
df_low_corr_features = pd.DataFrame({
    'Feature': X_cols[low_corr_features],
    'Max Correlation': row_max[low_corr_features]
}).sort_values(by='Max Correlation', ascending=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.write("Pairs with the highest correlation")
    st.dataframe(high_corr_pairs, use_container_width=True)
with col2:
    st.write("Most common features in the pairs")
    num_mcf = 6
    st.dataframe(pd.DataFrame({'Feature':[x[0] for x in most_common_features[:num_mcf]], 'Count': [x[1] for x in most_common_features[:num_mcf]]}), use_container_width=True)
with col3:
    st.write("Fearures with the lowest maximal correlation")
    st.dataframe(df_low_corr_features)

# display all the pairs that contain 'Median Age Mother'
st.write("### 'Mean Age of Mother total' appears often in the pairs with hight correlation")
search_term = 'Mean Age of Mother total'
mean_age_mother_pairs = df_corr_pairs[df_corr_pairs['Feature 1'].str.contains(search_term) | df_corr_pairs['Feature 2'].str.contains(search_term)]
mean_age_mother_pairs = mean_age_mother_pairs.reindex(mean_age_mother_pairs['Correlation'].abs().sort_values(ascending=False).index)
num_mean_age_mother_pairs = 8
col1, col2 = st.columns([2,3])
with col1:
    st.write(''); st.write('');
    st.write(''); st.write('');
    st.write('')
    st.dataframe(mean_age_mother_pairs[:num_mean_age_mother_pairs])

with col2:
    with st.spinner('Plotting the features...'):
# values = df[mean_age_mother_pairs['Feature 1'].iloc[:num_mean_age_mother_pairs]]
# doing it manual for more interesting features
        df_plot = df[['Year', 'Mean Age of Mother total', 'Inhabitants total', 'Income 25% total', 'Mean Age total', 'Income Median total', 'Income 75% total', 'Price Living Space total', 'Median Price total']]
# plot w/ plotly
        years = df['Year'].values
        df_plot['Year'] = years
# scaling
        scaling_factors = df_plot.drop(columns='Year').max()
        df_scaled = df_plot.copy()
        for col in scaling_factors.index:
            df_scaled[col] = df_scaled[col] / scaling_factors[col]
        df_melt = df_scaled.melt(id_vars='Year', var_name='Feature', value_name='Value')
        fig = px.line(df_melt, x='Year', y='Value', color='Feature', title="Features with the highest correlations with 'Mean Age of Mother total'")
        st.plotly_chart(fig, use_container_width=False)
        sep = ',' + '&nbsp;' * 4
        st.write(
            f'The features are scaled to the same range for better comparison with the following factors: <br>'
            f"{sep.join(f'{x:.2f}' for x in scaling_factors.values)}", unsafe_allow_html=True
)

# st.write("Why are 'Mean Age of Mother total' and 'Deaths total' correlated?")
st.write('We see that the age of the Mother, the prices for Appartments and the income are correlated.')
    



st.write("### Modeling with Trees and Forests")
st.write("To see which features are important to distinguish the Quartiers, we can use a regression tree or a random forest model and inspect the feature importances.")

with st.spinner('Training the models...'):
    y = df['Quartier'].values
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Decision Tree
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train, y_train)
    y_tree = tree.predict(X_test)
    acc_tree = accuracy_score(y_test, y_tree)
    f1_tree = f1_score(y_test, y_tree, average='weighted')

# Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_rf)
    f1_rf = f1_score(y_test, y_rf, average='weighted')


st.write(f'Fitting the Decision Tree model and the random forest model, we obained the following accuracy and F1 scores:')
st.dataframe(pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest'],
    'Accuracy': [acc_tree, acc_rf],
    'F1 Score': [f1_tree, f1_rf]
}), use_container_width=False)


    
tree_importances = pd.DataFrame({
    'Feature': X_cols.values,
    'Importance': tree.feature_importances_
}).sort_values(by='Importance', ascending=False)
forest_importances = pd.DataFrame({
    'Feature': X_cols.values,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

col1, col2 = st.columns(2)
with col1:
    st.write("Decision Tree feature importances:")
    st.dataframe(tree_importances)
with col2:
    st.write("Random Forest feature importances:")
    st.dataframe(forest_importances)

# compare which features are in the top 10 of both models
tree_top_10 = tree_importances['Feature'].iloc[:10].values
forest_top_10 = forest_importances['Feature'].iloc[:10].values
in_both = np.intersect1d(tree_top_10, forest_top_10)
df_in_both = pd.DataFrame({
    'Feature': in_both,
    'Tree Importance': tree_importances[tree_importances['Feature'].isin(in_both)]['Importance'].values,
    'Forest Importance': forest_importances[forest_importances['Feature'].isin(in_both)]['Importance'].values
}).sort_values(by='Tree Importance', ascending=False)
st.session_state['df_in_both'] = df_in_both # save for later use

df_in_both_melt = df_in_both.melt(id_vars='Feature', var_name='Model', value_name='Importance').sort_values(by='Importance', ascending=True)

st.write(f'#### The following features are in the top 10 of both models:')
fig = px.bar(
    df_in_both_melt,
    y='Feature',
    x='Importance',
    color='Model',
    orientation='h',
    barmode='group',
    title='Top Features in Both Models'
)
layout_toggle = st.toggle('Compact or wide layout', value=True)
if layout_toggle:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig)
else:
    st.plotly_chart(fig, use_container_width=True)



# i want to apply Isomap to the data and plot the results

st.write("In the following page, we cluster the Quartiers based on these features.")
st.page_link("pages/2_PCA_and_Clustering.py")
