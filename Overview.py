import numpy as np
import streamlit as st
import os
import pandas as pd
import plotly.express as px


import utilities as ut

st.set_page_config(layout="centered")




os.getcwd()
st.write('# Insights in to Zurich\'s Quartiers')
st.write('#### Through data analysis and visualization')

key_quartier_kreis = ut.load_q_k_key()
key_quartier_kreis['Kreis Num'] = pd.to_numeric(key_quartier_kreis['Kreis'].apply(lambda x: int(x.split(' ')[-1])), downcast='unsigned')


st.write('Some first impressions of the data')
# plot w/ plotly.express the data 'Mean Age Total', 'Deaths total'
df = ut.load_dataframe()
# st.dataframe(df)

fig = px.line(df, x='Year', y='Mean Age total')
fig.update_traces(name='Mean Age total', showlegend=True)
fig.add_scatter(x=df['Year'], y=df['Inhabitants total']/1e4, mode='lines', name='Inhabitants total per 10k')
fig.add_scatter(x=df['Year'], y=df['Num Deaths total']/1e2, mode='lines', name='Deaths total per 100')
fig.add_scatter(x=df['Year'], y=df['Num Births total']/1e2, mode='lines', name='Births total per 100')
fig.add_scatter(x=df['Year'], y=df['Income Median total']/1e2, mode='lines', name='Median Income total per 100')
fig.add_scatter(x=df['Year'], y=df['Median Price total']/1e7, mode='lines', name='Median Price total per 10 M')
fig.update_layout(title='Some overal developments in Zurich',
                    xaxis_title='Year',
                    yaxis_title='Data',
                    legend_title='Data',
                  )
st.plotly_chart(fig, use_container_width=True)



st.write('#### The following features are available in the data:')
st.dataframe(pd.DataFrame(df.columns, columns=["Available Features"]))
st.markdown('''
            The data can be obtained from the [Zürich Open Data](https://data.stadt-zuerich.ch/) website.
            ''')


ut.load_X_scaled()  # Load the scaled data to ensure it's cached
