import streamlit as st
import plotly as py
import numpy as np
import pandas as pd
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data_2k15 = pd.read_csv('2015.csv')
data_2k16 = pd.read_csv('2016.csv')
data_2k17 = pd.read_csv('2017.csv')
data_2k18 = pd.read_csv('2018.csv')
data_2k19 = pd.read_csv('2019.csv')
data_2k20 = pd.read_csv('2020.csv')
data_2k21 = pd.read_csv('world-happiness-report-2021.csv')

data = dict(
        type = 'choropleth',
        marker_line_width=1,
        locations = data_2k15['Country'],
        locationmode = "country names",
        z = data_2k21['Ladder score'],
        text = data_2k15['Country'],
        colorbar = {'title' : 'Happiness score'})
layout = dict(#title = 'Happiness Map for the year 2021',
                  geo = dict(projection = {'type':'mercator'})
                 )
choromap = go.Figure(data = [data],layout = layout)

df2015 = data_2k15.iloc[:30,:]
df2016 = data_2k16.iloc[:30,:]
df2017 = data_2k17.iloc[:30,:]
df2018 = data_2k18.iloc[:30,:]
df2019 = data_2k19.iloc[:30,:]
df2020 = data_2k20.iloc[:30,:]
df2021 = data_2k21.iloc[:30,:]

# Creating curve1
curve1 = go.Scatter(x = df2015['Country'],
                    y = df2015['Happiness Score'],
                    mode = "lines+markers",
                    name = "2015",
                    marker = dict(color = 'red'),
                    text= df2015.Country)

# Creating curve2
curve2 = go.Scatter(x = df2015['Country'],
                    y = df2016['Happiness Score'],
                    mode = "lines+markers",
                    name = "2016",
                    marker = dict(color = 'blue'),
                    text= df2015.Country)

# Creating curve3
curve3 = go.Scatter(x = df2015['Country'],
                    y = df2017['Happiness.Score'],
                    mode = "lines+markers",
                    name = "2017",
                    marker = dict(color = 'green'),
                    text= df2015.Country)

# Creating curve4
curve4 = go.Scatter(x = df2015['Country'],
                    y = df2018['Score'],
                    mode = "lines+markers",
                    name = "2018",
                    marker = dict(color = 'black'),
                    text= df2015.Country)

# Creating curve5
curve5 = go.Scatter(x = df2015['Country'],
                    y = df2019['Score'],
                    mode = "lines+markers",
                    name = "2019",
                    marker = dict(color = 'pink'),
                    text= df2015.Country)


# Creating curve6
curve6 = go.Scatter(x = df2015['Country'],
                    y = df2020['Ladder score'],
                    mode = "lines+markers",
                    name = "2020",
                    marker = dict(color = 'purple'),
                    text= df2015.Country)

# Creating curve7
curve7 = go.Scatter(x = df2015['Country'],
                    y = df2021['Ladder score'],
                    mode = "lines+markers",
                    name = "2021",
                    marker = dict(color = 'orange'),
                    text= df2015.Country)


data = [curve1, curve2, curve3, curve4, curve5,curve6,curve7]
layout = dict(#title = 'Top 30 countries according to their happiness score from 2015 to 2021',
              xaxis= dict(title= 'Countries'),
              yaxis= dict(title= 'Happiness Score'),
              hovermode="x unified"
             )

fig = dict(data = data, layout = layout)

st.title("World happiness report analysis")
st.write('### Happiness Map for the year 2021')
st.plotly_chart(choromap)
st.write("### Top 30 countries according to their happiness score from 2015 to 2021")
st.plotly_chart(fig)

data_2k15['Year']='2015'
data_2k16['Year']='2016'
data_2k17['Year']='2017'
data_2k18['Year']='2018'
data_2k19['Year']='2019'
data_2k20['Year']='2020'
data_2k21['Year']='2021'

data_2k15.rename(columns={'Economy (GDP per Capita)':'GDP per capita'},inplace=True)
data1=data_2k15.filter(['Country','GDP per capita',"Year"],axis=1)

data_2k16.rename(columns={'Economy (GDP per Capita)':'GDP per capita'},inplace=True)
data2=data_2k16.filter(['Country','GDP per capita',"Year"],axis=1)

data_2k17.rename(columns={'Economy..GDP.per.Capita.':'GDP per capita'},inplace=True)
data3=data_2k17.filter(['Country','GDP per capita','Year'],axis=1)

data_2k18.rename(columns={'Country or region':'Country'},inplace=True)
data4=data_2k18.filter(['Country','GDP per capita',"Year"],axis=1)

data_2k19.rename(columns={'Country or region':'Country'},inplace=True)
data5=data_2k19.filter(['Country','GDP per capita','Year'],axis=1)

data_2k20.rename(columns={'Country name':'Country','Explained by: Log GDP per capita':'GDP per capita'},inplace=True)
data6=data_2k20.filter(['Country','GDP per capita','Year'],axis=1)

data_2k21.rename(columns={'Country name':'Country','Explained by: Log GDP per capita':'GDP per capita'},inplace=True)
data7=data_2k21.filter(['Country','GDP per capita','Year'],axis=1)

data1=data1.append([data2,data3,data4,data5,data6,data7])

plt.figure(figsize=(10,8))
df = data1[data1['Country']=='India']
sns.lineplot(x="Year", y="GDP per capita",data=df,label='India')

df = data1[data1['Country']=='United States']
sns.lineplot(x="Year", y="GDP per capita",data=df,label='US')

df = data1[data1['Country']=='Finland']
sns.lineplot(x="Year", y="GDP per capita",data=df,label='Finland')

df = data1[data1['Country']=='United Kingdom']
sns.lineplot(x="Year", y="GDP per capita",data=df,label="UK")

df = data1[data1['Country']=='Canada']
sns.lineplot(x="Year", y="GDP per capita",data=df,label='Canada')

df = data1[data1['Country']=='Switzerland']
sns.lineplot(x="Year", y="GDP per capita",data=df,label='Switzerland')

df = data1[data1['Country']=='United Arab Emirates']
sns.lineplot(x="Year", y="GDP per capita",data=df,label='United Arab Emirates')

df = data1[data1['Country']=='Pakistan']
sns.lineplot(x="Year", y="GDP per capita",data=df,label='Pakistan')

df = data1[data1['Country']=='Afghanistan']
sns.lineplot(x="Year", y="GDP per capita",data=df,label='Afghanistan')

df = data1[data1['Country']=='Australia']
sns.lineplot(x="Year", y="GDP per capita",data=df,label='Australia')

df = data1[data1['Country']=='China']
sns.lineplot(x="Year", y="GDP per capita",data=df,label='China')

df = data1[data1['Country']=='Russia']
sns.lineplot(x="Year", y="GDP per capita",data=df,label='Russia')

df = data1[data1['Country']=='France']
sns.lineplot(x="Year", y="GDP per capita",data=df,label='France')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
st.write("### GDP per capita of some countries over the years")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

dcr =data_2k21.drop(['Standard error of ladder score', 'upperwhisker', 'lowerwhisker','Ladder score in Dystopia',
       'Explained by: Social support','Logged GDP per capita',
       'Explained by: Healthy life expectancy',
       'Explained by: Freedom to make life choices',
       'Explained by: Generosity', 'Explained by: Perceptions of corruption','Dystopia + residual'],axis=1)
cor = dcr.corr() #Calculate the correlation of the above variables

st.write("This heatmap represents the correlation between various features of the data.It can be seen that the ladder score "
         "i.e the happiness score mostly depends on features like 'GDP per capita,social support,healthy life expectancy and "
         "freedom to make life choices'.It is least correlated with ''generosity' and 'Perception of corruption'.")

sns.heatmap(cor, square = True) #Plot the correlation as heat map

st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


st.markdown("### According to the world happiness report of 2021:")
st.write("- Happiest Country - Finland")
st.write("- Highest GDP Per Capita (Economy) - Luxembourg")
st.write("- Highest Social Support - Turkmenistan and Iceland")
st.write("- High Healthy Life Expectancy Country - Singapore")
st.write("- Highest score in freedom to make life choices - Norway and Uzbekistan")
st.write("- Highest Perception of Corruption - Singapore")
st.write("- Highest Generosity - Indonesia")

st.sidebar.write("Countries has been clustered on the basis of eight parameters namely, happiness score,"
                 " GDP per capita, social support,life expectancy, freedom, generosity, corruption and Dystopia residual ")

clustering_features = st.sidebar.selectbox("Visualize clusters on the basis of",("Happiness score and GDP per capita","Happiness score and healthy life expectancy",
                                            "Happiness score and freedom to make life choices","Happiness score and social support",
                                            "GDP per capita and healthy life expectancy","GDP per capita and social support"))

st.write("### Visualizing clusters on the basis of: ")
st.write(clustering_features)

data = data_2k21[['Ladder score','GDP per capita', 'Explained by: Social support',
       'Explained by: Healthy life expectancy',
       'Explained by: Freedom to make life choices',
       'Explained by: Generosity', 'Explained by: Perceptions of corruption','Dystopia + residual']]

ss = StandardScaler()

def Kmeans_clst(X, nclust=2):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_
    return (clust_labels, cent)

clust_labels, cent = Kmeans_clst(data, 2)
kmeans = pd.DataFrame(clust_labels)
data.insert((data.shape[1]),'kmeans',kmeans)

if clustering_features=="Happiness score and GDP per capita":
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data['GDP per capita'], data['Ladder score'], c=kmeans[0])
    ax.set_title('K-Means Clustering')
    ax.set_xlabel('GDP per Capita')
    ax.set_ylabel('Happiness score')
    plt.colorbar(scatter)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

elif clustering_features=="Happiness score and healthy life expectancy":
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data['Explained by: Healthy life expectancy'], data['Ladder score'], c=kmeans[0])
    ax.set_title('K-Means Clustering')
    ax.set_xlabel('Healthy life expectancy')
    ax.set_ylabel('Happiness score')
    plt.colorbar(scatter)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

elif clustering_features=="Happiness score and freedom to make life choices":
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data['Explained by: Freedom to make life choices'], data['Ladder score'], c=kmeans[0])
    ax.set_title('K-Means Clustering')
    ax.set_xlabel('Freedom to decisions')
    ax.set_ylabel('Happiness score')
    plt.colorbar(scatter)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


elif clustering_features=="Happiness score and social support":
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data['Explained by: Social support'], data['Ladder score'], c=kmeans[0])
    ax.set_title('K-Means Clustering')
    ax.set_xlabel('Social support')
    ax.set_ylabel('Happiness score')
    plt.colorbar(scatter)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

elif clustering_features=="GDP per capita and healthy life expectancy":
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data['Explained by: Healthy life expectancy'], data['GDP per capita'], c=kmeans[0])
    ax.set_title('K-Means Clustering')
    ax.set_ylabel('GDP per Capita')
    ax.set_xlabel('Healthy life expectancy')
    plt.colorbar(scatter)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

elif clustering_features=="GDP per capita and social support":
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data['Explained by: Social support'], data['GDP per capita'], c=kmeans[0])
    ax.set_title('K-Means Clustering')
    ax.set_ylabel('GDP per Capita')
    ax.set_xlabel('Social support')
    plt.colorbar(scatter)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

