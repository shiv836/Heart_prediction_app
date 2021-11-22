# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import streamlit as st
import pandas as pd
import numpy as np
st.write("Minor project 2021")
col1, mid, col2 = st.columns([1,1,20])
with col1:
    st.image('nit_raipur_logo.jpg', width=60)
with col2:
    st.write('NIT Raipur')

st.write("""
# Heart disease Prediction App
This app predicts the **Probability of heart disease**!
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        age=st.sidebar.slider('Age',1,100,30)
        sex = st.sidebar.selectbox('Sex',('male','female'))
        cp=st.sidebar.selectbox('ChestPain-type',('typical angina','atypical angina','non-anginal pain',' asymptomatic'))
        trestbps=st.sidebar.slider('resting-blood-pressure',1,250,140)
        chol=st.sidebar.slider('cholestrol',1,600,140)
        fbs = st.sidebar.selectbox('Is your Fasting Blood sugar greater than 120?', ('Yes', 'No'))
        restECG=st.sidebar.selectbox('resting electrocardiographic results',('Normal','ST-T Wave abnormality','Hypertrophy'))
        thalach=st.sidebar.slider('Maximum-Heart-Rate',1,80,200)
        exang=st.sidebar.selectbox('Do you have exercise-induced angina', ('Yes', 'No'))
        oldPeak=st.sidebar.slider(' ST depression induced by exercise relative to rest',1,4,10)
        slope=st.sidebar.selectbox('the slope of the peak exercise ST segment',('upsloping','flat','downsloping'))
        ca=st.sidebar.selectbox('Number of vessels coloured by fluoroscopy',('0','1','2','3','4'))
        thal=st.sidebar.selectbox('thal',('Normal','fixed defect','reversable defect'))

        if(sex=="male"):
            sex=1.0
        else:
            sex=0.0

        if(cp=="typical angina"):
            cp=1.0
        elif(cp=="atypical angina"):
            cp=2.0
        elif(cp=="non-anginal pain"):
            cp=3.0
        else:
            cp=0.0

        if(fbs=="Yes"):
            fbs=1.0
        else:
            fbs=0.0

        if (restECG == "Normal"):

            restECG= 0.0
        elif (cp == "atypical angina"):
            restECG = 1.0
        else:
            restECG = 2.0

        if(exang=="Yes"):
            exang=1.0
        else:
            exang=0.0
        if(slope=="upsloping"):
            slope=1.0
        elif(slope=="flat"):
            slope=2.0
        else:
            slope=3.0
        if(ca=="0"):
            ca=0.0
        elif(ca=="1"):
            ca=1.0
        elif(ca=="2"):
            ca=2.0
        elif(ca=="3"):
            ca=3.0
        else:
            ca=4.0
        if(thal=="Normal")  :
            thal=1.0
        elif(thal=="fixed defect"):
            thal=2.0
        else:
            thal=3.0
        data = {'age': age,
                'sex': sex,
                'cp': cp,
                'trestbps': trestbps,
                'chol': chol,
                'fbs': fbs,
                'restecg':restECG,
                'thalach':thalach,
                'exang':exang,
                'oldpeak': oldPeak,
                'slope':slope,
                'ca':ca,
                'thal':thal}


        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()
display=input_df
# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
heart_raw = pd.read_csv('heart.csv')
heart = heart_raw.drop(columns=['target'])
df = pd.concat([input_df,heart],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
# encode = ['sex','island']
# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix=col)
#     df = pd.concat([df,dummy], axis=1)
#     del df[col]
from sklearn.preprocessing import StandardScaler
df = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
df[columns_to_scale] = standardScaler.fit_transform(df[columns_to_scale])
df.fillna(df['oldpeak'].mean())
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(display)

# Reads in saved classification model
import pandas as pd
# import seaborn as sns
# from matplotlib.pyplot import clf
from sklearn.model_selection import train_test_split
# sns.set()

heart = pd.read_csv('heart.csv')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
data = heart.copy()
from sklearn.preprocessing import StandardScaler
data = pd.get_dummies(data, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
data[columns_to_scale] = standardScaler.fit_transform(data[columns_to_scale])

# Separating X and y
y = data['target']
X = data.drop(['target'], axis = 1)
X.fillna(X.mean(), inplace=True)
# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Apply model to make predictions
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)


st.subheader('Prediction')
penguins_species = np.array(['YES','NO'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)