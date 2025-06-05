import plotly.graph_objects as go
import streamlit as st

quart_shown = st.selectbox("Select Quartier", quarlangs)


# buttons = [
#     dict(
#         label=ql,
#         method="update",
#         args=[
#             {"x": [df_bev_u_nat['StichtagDatJahr'][df_bev_u_nat['QuarLang'] == ql]],
#              "y": [df_bev_u_nat['AnzBestWir'][df_bev_u_nat['QuarLang'] == ql]]}
#         ]
#     )
#     for ql in quarlangs
# ]
# fig_test.update_layout(
#     updatemenus=[
#         dict(
#             buttons=buttons,
#             direction="down",
#             showactive=True,
#             x=0.1,
#             xanchor="left",
#             y=1.15,
#             yanchor="top"
#         )
#     ]
# )

fig_test = go.Figure()
# inhabitants
fig_test.add_trace(go.Scatter(
    x=df_bev_u_nat['StichtagDatJahr'][df_bev_u_nat['QuarLang'] == quart_shown],
    y=df_bev_u_nat['AnzBestWir'][df_bev_u_nat['QuarLang'] == quart_shown],
    mode='markers+lines',
    name=f"{quart_shown} bevölkerung"
))
# nationalities
fig_test.add_trace(go.Scatter(
    x=df_bev_u_nat['StichtagDatJahr'][df_bev_u_nat['QuarLang'] == quart_shown],
    y=df_bev_u_nat['AnzNat'][df_bev_u_nat['QuarLang'] == quart_shown],
    mode='markers+lines',
    name=f"{quart_shown} num nationalitäten"
))
# social assistance
fig_test.add_trace(go.Scatter(
    x=df_sozialhilfe['Jahr'][df_sozialhilfe['Raum'] == quart_shown],
    y=df_sozialhilfe['SH_Quote'][df_sozialhilfe['Raum'] == quart_shown],
    mode='markers+lines',
    name=f"{quart_shown} sozialhilfe quote in %"
))
# leerflaechen
fig_test.add_trace(go.Scatter(
    x=df_leer_sum['StichtagDatJahr'][df_leer_sum['QuarLang'] == quart_shown],
    y=df_leer_sum['Leerflaeche'][df_leer_sum['QuarLang'] == quart_shown],
    mode='markers+lines',
    name=f"{quart_shown} leerflaechen"
))
# motorisierungsgrad
fig_test.add_trace(go.Scatter(
    x=df_motorisierungsgrad['StichtagDatJahr'][df_motorisierungsgrad['QuarLang'] == quart_shown],
    y=df_motorisierungsgrad['Motorisierungsgrad'][df_motorisierungsgrad['QuarLang'] == quart_shown],
    mode='markers+lines',
    name=f"{quart_shown} motorisierungsgrad"
))
# neuzulassungen
fig_test.add_trace(go.Scatter(
    x=df_neuzulassungen_sum['StichtagDatJahr'][df_neuzulassungen_sum['QuarLang'] == quart_shown],
    y=df_neuzulassungen_sum['FzAnz'][df_neuzulassungen_sum['QuarLang'] == quart_shown],
    mode='markers+lines',
    name=f"{quart_shown} neuzulassungen"
))
# pkw total
fig_test.add_trace(go.Scatter(
    x=df_pkw_sum['StichtagDatJahr'][df_pkw_sum['QuarLang'] == quart_shown],
    y=df_pkw_sum['FzAnz'][df_pkw_sum['QuarLang'] == quart_shown],
    mode='markers+lines',
    name=f"{quart_shown} pkw total"
))
# todesfaelle
fig_test.add_trace(go.Scatter(
    x=df_todesfaelle['StichtagDatJahr'][df_todesfaelle['StichtagDatJahr'] == quart_shown],
    y=df_todesfaelle['AnzSterWir'][df_todesfaelle['StichtagDatJahr'] == quart_shown],
    mode='markers+lines',
    name=f"{quart_shown} todesfälle"
))
    

fig_test.update_layout(
    title='Comparison of Quartier Data',
    xaxis_title='Year',
    yaxis_title='Numbers'
)
st.plotly_chart(fig_test)
