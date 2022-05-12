import pandas as pd
from functools import partial, reduce
import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px
import numpy as np
import streamlit as st

with st.echo(code_location='below'):
    xls=pd.read_excel('Museums Data_f.xlsx',sheet_name=None)

    list_index=['Visitor Figures','Under 16 visitor figures',
    'Overseas visitor figures','Number of unique website visits',
    'Proportion of visitors who would recommend a visit',
    'Facilitated and self-directed visits by visitors under 18 years old and in formal education',
    'The number of instances where visitors under 18 years old have participated in on-site activities',
    'Number of UK loan venues',
    'Self-generated income: Admissions',
    'Self-generated income: Trading income (net profit)',
    'Self-generated income: Fundraising, split by museum']

    #Сделаем словарь с датафреймами из листов файла Excel
    for i in range(1,len(xls)+1):
        b={}
        for j in list(xls[f'Table {i}'].columns[0:]):
            if j!='Museum/Gallery':
                b[j]=list_index[i-1]
            else:
                b[j]='Museum/Gallery'
        xls[f'Table {i}'].columns = pd.MultiIndex.from_tuples([*zip(map(b.get,xls[f'Table {i}']), xls[f'Table {i}'])])
        xls[f'Table {i}'].set_index(xls[f'Table {i}'].columns[0],inplace=True)
        #FROM: https://stackoverflow.com/questions/53714192/pandas-dataframe-how-to-group-columns-together-in-python#

    #Настроим Multiindex и сконкатинируем листы в df_fin
    my_reduce = partial(pd.merge, left_index=True,right_index=True, how='outer')
    df=reduce(my_reduce, xls.values())
    #https://stackoverflow.com/questions/53935848/how-to-merge-all-data-frames-in-a-dictionary-in-python#
    df.index.name='Museum'
    df_fin=df.loc[df.index.dropna()]

    idx=pd.IndexSlice

    #Dataset для Heatmap
    dict_mean={}
    for i in list_index:
        dict_mean[f'{i}']=pd.DataFrame(df_fin.iloc[:,df_fin.columns.get_level_values(0)==f'{i}'].mean(axis=1)).dropna()
        dict_mean[f'{i}']=dict_mean[f'{i}'].rename(columns={dict_mean[f'{i}'].columns[0]:f'{i}'})
    my_reduce_m = partial(pd.merge, left_index=True,right_index=True, how='outer')
    df_m=reduce(my_reduce_m, dict_mean.values())
    df_m=df_m.dropna()

    #Составим Dataset для Parallel Coordinates
    list_years=['2008/2009','2009/2010','2011/2012','2012/2013','2013/2014','2014/2015','2015/2016','2017/2018','2018/2019']
    institution_codes={'Museums of Fine Arts':1,'Museums of Modern Art':2,'Historical Museums':3,'Other':4}

    dict_prop0={}
    for i in list_years:
        dict_prop0[f'{i}']=df_fin.loc[:,idx['Under 16 visitor figures',f'{i}']].dropna().div(df_fin.loc[:,idx['Visitor Figures',f'{i}']].dropna(),axis='index')
        dict_prop0[f'{i}']=dict_prop0[f'{i}'].rename(f'Proportion of visiors under 16 visitors {i}')
    my_reduce_prop = partial(pd.merge, left_index=True,right_index=True, how='outer')
    df_prop0=reduce(my_reduce_prop, dict_prop0.values())
    df_prop0=df_prop0.dropna()
    df_prop0['Proportion of visitors under 16']=df_prop0.mean(axis=1)

    dict_prop1={}
    for i in list_years[1:]:
        dict_prop1[f'{i}']=df_fin.loc[:,idx['Visitor Figures',f'{i}']].dropna().div(df_fin.loc[:,idx['Number of unique website visits',f'{i}']].dropna(),axis='index')
        dict_prop1[f'{i}']=dict_prop1[f'{i}'].rename(f'Visitors conversion rate {i}')
    my_reduce_prop = partial(pd.merge, left_index=True,right_index=True, how='outer')
    df_prop1=reduce(my_reduce_prop, dict_prop1.values())
    df_prop1=df_prop1.dropna()
    df_prop1['Visitors conversion rate']=df_prop1.mean(axis=1)

    dict_prop2={}
    for i in list_years:
        dict_prop2[f'{i}']=df_fin.loc[:,idx['Overseas visitor figures',f'{i}']].dropna().div(df_fin.loc[:,idx['Visitor Figures',f'{i}']].dropna(),axis='index')
        dict_prop2[f'{i}']=dict_prop2[f'{i}'].rename(f'Proportion of overseas visitors {i}')
    my_reduce_prop = partial(pd.merge, left_index=True,right_index=True, how='outer')
    df_prop2=reduce(my_reduce_prop, dict_prop2.values())
    df_prop2=df_prop2.dropna()
    df_prop2['Proportion of overseas visitors']=df_prop2.mean(axis=1)

    dict_prop3={}
    for i in list_years[3:]:
        dict_prop3[f'{i}']=df_fin.loc[:,idx['Facilitated and self-directed visits by visitors under 18 years old and in formal education',f'{i}']].dropna().div(df_fin.loc[:,idx['Visitor Figures',f'{i}']].dropna(),axis='index')
        dict_prop3[f'{i}']=dict_prop3[f'{i}'].rename(f'Proportion of formal education visits {i}')
    my_reduce_prop = partial(pd.merge, left_index=True,right_index=True, how='outer')
    df_prop3=reduce(my_reduce_prop, dict_prop3.values())
    df_prop3=df_prop3.dropna()
    df_prop3['Proportion of formal education visits']=df_prop3.mean(axis=1)

    df_prop=pd.concat([pd.DataFrame(df_prop0['Proportion of visitors under 16']),pd.DataFrame(df_prop1['Visitors conversion rate']),pd.DataFrame(df_prop2['Proportion of overseas visitors']),pd.DataFrame(df_prop3['Proportion of formal education visits'])],axis=1)
    df_prop=df_prop.dropna()
    institution_code=[3,3,1,4,1,3,3,4,4,1,2,4,1]
    df_prop['Institution Code']=institution_code
    list_codes=list(df_prop['Institution Code'].unique().astype(str))
    
    st.title("Визуализация датасета о деятельности Музеев Великобритании")
    
    st.header("Распределение основных показателей")
    #BoxPlot
    fig=go.Figure()
    for dataset in list_index:
        fig.add_trace(go.Box(x=pd.melt(df_fin.loc[:,idx[f'{dataset}',:]])['variable_1'],y=pd.melt(df_fin.loc[:,idx[f'{dataset}',:]])['value'],name=dataset))
    buttons=[]
    ### FROM: https://docs.datapane.com/examples-and-tutorials/interactive-filters#plotly
    for i,dataset in enumerate(list_index):
        args = [False] * len(list_index)
        args[i] = True
        button = dict(label = dataset,
                      method = "update",
                      args=[{"visible": args}])
        buttons.append(button)
    fig.update_layout(
        updatemenus=[dict(
                        active=0,
                        type="dropdown",
                        buttons=buttons,
                        x = 0,
                        y = 1,
                        xanchor = 'left',
                        yanchor = 'bottom'
                    )],
        autosize=False,
        width=1000,
        height=800)
    st.plotly_chart(fig)
    ### END FROM
    st.subheader("Данные визуализованы в виде boxplot графиков, построенных в соответсвии с годовыми наблюдениями и расстортированных по типу данных") 

    st.header("Корреляция между основыми показателями музейной деятельности")
    #Heatmap
    df_m_corr=df_m.corr()
    sns.set(rc = {'figure.figsize':(15,8)})
    #how to mask an upper triangle: https://python-graph-gallery.com/90-heatmaps-with-various-input-format#
    mask = np.zeros_like(df_m_corr)
    mask[np.triu_indices_from(mask)] = True
    heat=sns.heatmap(df_m_corr,annot=True,xticklabels=False,mask=mask)
    heat.set_facecolor('xkcd:plum')
    fig_h=heat.get_figure()
    st.pyplot(fig_h)
    st.subheader("Данные для матрицы были усреднены - с 2008 до 2019 г")
    
    st.header("Мультипликаторы Музейной Деятельности")

    #Parallel Coordinates
    ### FROM: https://stackoverflow.com/questions/64100889/add-dropdown-menu-to-plotly-express-treemap
    traces = []
    buttons = []
    for i, dataset in enumerate(list_codes):
        args = [False] * len(list_codes)
        args[i] = True
        traces.append(
            px.parallel_coordinates(df_prop[df_prop['Institution Code'] == int(dataset)], color='Institution Code',
                                    dimensions=list(df_prop[df_prop['Institution Code'] == int(dataset)].columns),
                                    color_continuous_scale=px.colors.diverging.Tealrose).update_traces(
                visible=True if i == 0 else False).data[0])
        button = dict(label=list(institution_codes.keys())[list(institution_codes.values()).index(int(dataset))],
                      method="update",
                      args=[{"visible": args}])
        buttons.append(button) ###END FROM:
    updatemenus = [dict(
        active=0,
        type="dropdown",
        buttons=buttons,
        x=0,
        y=1.25,
        xanchor='left',
        yanchor='bottom',
        direction='right'
    )]
    fig_par1 = go.Figure(data=traces, layout=dict(updatemenus=updatemenus))
    fig_par1.update_layout(width=900)
    st.plotly_chart(fig_par1)
    
    st.subheader("Данные сгруппированы по одному из 4 типов музейных институций")











