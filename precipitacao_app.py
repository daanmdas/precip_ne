import pandas as pd
import numpy as np
import datetime as dt
from PIL import Image
import plotly.graph_objects as go
import streamlit as st
import folium
import plotly.express as px
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static

im = Image.open("raining.png")
st.set_page_config(layout='wide',
                   page_title = 'Precipitação',
                   page_icon = im)

@st.cache_data()
def param():
        
    parametros = pd.read_csv("https://raw.githubusercontent.com/daanmdas/precip_ne/main/parametros.csv")
    
    return parametros

parametros = param()

@st.cache_data()
def dados_parametrizados(dados, retorno = 10):
  
    df = dados.loc[dados['Retorno'] == retorno]
    inicial = df.loc[df['Mes'] == dt.datetime.now().month]
  
    return inicial, df

dados = pd.read_csv('https://raw.githubusercontent.com/daanmdas/precip_ne/main/retorno.csv')

retorno = int(st.sidebar.text_input(label = "Periodo de Retorno (Anos)", value= 10))

df = dados.loc[dados['Retorno'] == retorno]
inicial = df.loc[df['Mes'] == dt.datetime.now().month]

if retorno != 10:

    inicial, df = dados_parametrizados(dados, retorno)

uf = parametros['UF'].unique()
cidades = parametros['cid'].unique()

info_sidebar = st.sidebar.empty()
st.sidebar.subheader("Tabela de Dados")
table = st.sidebar.empty()

uf = np.append(uf, 'Todos')
inx_uf = np.where(uf == 'Todos')[0].max()

st.title("Precipitação Extrema Nordeste Brasileiro")

escolha_uf = st.sidebar.selectbox(
    label = "Escolha o Estado",
    options = uf,
    index= int(inx_uf)
    )

if escolha_uf == 'Todos':

    uf_filtrados = inicial.copy()
    cidades = uf_filtrados['Cidade'].unique()
    cidades = np.append(cidades, 'Todas')
    inx_cidades = np.where(cidades == 'Todas')[0].max() 

else: 
    uf_filtrados = inicial.loc[inicial['UF'] == escolha_uf]
    cidades = uf_filtrados.loc[uf_filtrados['UF'] == escolha_uf]['Cidade'].unique()
    cidades = np.append(cidades, 'Todas')
    inx_cidades = np.where(cidades == 'Todas')[0].max()

escolha_cidade = st.sidebar.selectbox(
    label = "Escolha a cidade",
    options = cidades,
    index= int(inx_cidades)
    )

if escolha_cidade == 'Todas':

    dados_filtrados = uf_filtrados.copy()
else:
    dados_filtrados = uf_filtrados.loc[uf_filtrados['Cidade'] == escolha_cidade]

aba1, aba2, aba3 = st.tabs([':earth_americas: Mapa', ':rain_cloud: Dados', ':chart_with_upwards_trend: Meses'])

with aba1:

    marker_map = folium.Map(location=[dados_filtrados['lat'].mean(), dados_filtrados['lon'].mean()],
                            tiles='cartodbpositron',
                            zoom_start=5
                            )

    marker_cluster = MarkerCluster().add_to(marker_map)
    for name, row in dados_filtrados.iterrows():
        pop = '''{0} <br>Precipitação Máx: {1}'''.format(row['chave'], round(row['Predito'],2))
        folium.Marker([row['lat'], row['lon']], 
                    popup = pop
                    ).add_to(marker_cluster)

    folium_static(marker_map, width=900, height=400)

with aba2:

    st.dataframe(dados_filtrados[['Cidade', 'UF', 'Mes',
                                  'Limite Inferior', 'Predito',
                                  'Limite Superior']].sort_values('Predito',
                                                                  ascending = False).reset_index(drop=True))

with aba3:


    if escolha_cidade != 'Todas':

        df_filtrados = df.loc[df['Cidade'] == escolha_cidade].set_index('Mes')
        fig = go.Figure()
        fig.update_layout(
            title='Precipitação Extrema Modelo GEV',
            xaxis=dict(title='Mes'),
            yaxis=dict(title='Precipitacao'),
        )

        # Adicionar hovertext personalizado
        #hover_text = ['{} mm'.format(precipitacao) for precipitacao in df_filtrados['mediaqestrela']]

        # Plotar a série temporal medianqu95 com pontos e hovertext
        fig.add_trace(go.Scatter(x=df_filtrados.index, y=df_filtrados['Predito'],
                                mode='lines+markers', line=dict(color='grey'), name='Precipitação Predita'))
                                #, hovertext=hover_text))

        # Plotar a série temporal lsqu95
        fig.add_trace(go.Scatter(x=df_filtrados.index, y=df_filtrados['Limite Superior'],
                                mode='lines', line=dict(color='blue', dash='dash'), name='Limite Superior'))

        # Plotar a série temporal liqu95
        fig.add_trace(go.Scatter(x=df_filtrados.index, y=df_filtrados['Limite Inferior'],
                                mode='lines', line=dict(color='red', dash='dash'), name='Limite Inferior'))

        # Substituir os rótulos do eixo x
        meses = ['1 - Jan', '2 - Fev', '3 - Mar', '4 - Abr', '5 - Mai', '6 - Jun',
                '7 - Jul', '8 - Ago', '9 - Set', '10 - Out', '11 - Nov', '12 - Dez']
        fig.update_layout(xaxis_tickvals=np.arange(1, 13, 1),
                        xaxis_ticktext=meses)
        
        st.plotly_chart(fig, use_container_width=True)

        prob = st.sidebar.checkbox('Calculo de Probabilidade de Precipitação')
        if prob:

            try:
                valor = float((st.sidebar.text_input(label = "Volume Precipitação (mm): ")).replace(',', '.'))
                mes = int(st.sidebar.text_input(label = 'Mês: '))
    
                w1 = np.cos((2 * np.pi * mes) / 12)
                w2 = np.sin((2 * np.pi * mes ) / 12)
                qu = (parametros.loc[parametros['cid'] == escolha_cidade]['beta0qumcb'] + 
                    parametros.loc[parametros['cid'] == escolha_cidade]['beta1qumcb'] * w1 + 
                    parametros.loc[parametros['cid'] == escolha_cidade]['beta2qumcb'] * w2)

                sigma = np.exp(parametros.loc[parametros['cid'] == escolha_cidade]['beta0sigmamcb'] + 
                            parametros.loc[parametros['cid'] == escolha_cidade]['beta1sigmamcb'] * w1 + 
                            parametros.loc[parametros['cid'] == escolha_cidade]['beta2sigmamcb'] * w2)

                xi = np.exp(parametros.loc[parametros['cid'] == escolha_cidade]['beta0ximcb'] + 
                            parametros.loc[parametros['cid'] == escolha_cidade]['beta1ximcb'] * w1 + 
                            parametros.loc[parametros['cid'] == escolha_cidade]['beta2ximcb'] * w2) - 1

                pestrela_calc = np.exp(-(((-np.log(0.99)) ** -xi ) + (xi * ((valor - qu) / sigma))) ** (-1 / xi))
                pestrela_calc = np.quantile(pestrela_calc, 0.5)
                retorno = 1 / (1 - pestrela_calc)
                st.sidebar.metric('Periodo de Retorno (Anos)',  int(round(retorno,0)), f'{round((1 - pestrela_calc) * 100, 4)} %')

            except:
                st.error('Preencha as informações necessárias ao lado')
            
    else:

        st.text('Escolha uma cidade para visualizar a previsão de precipitação ao longo do ano')
