import streamlit as st
import pandas as pd
import joblib
from PIL import Image

model = open("Knn_Classifier.pkl","rb")
model = joblib.load(model)

st.title("Iris flower species Classification App")

setosa= Image.open("setosa.jpg")
versicolor= Image.open('versiclor.jpg')
virginica = Image.open('virginia.jpg')

virginica = virginica.resize((500,500))
setosa = setosa.resize((500,500))
versicolor = versicolor.resize((500,500))


st.sidebar.title("Features")

parameter_list=['Sepal length (cm)','Sepal Width (cm)','Petal length (cm)','Petal Width (cm)']
parameter_input_values=[]
parameter_default_values=['5.2','3.2','4.2','1.2']
values=[]


#Display
for parameter, parameter_df in zip(parameter_list, parameter_default_values):
 
 values= st.sidebar.slider(label=parameter, key=parameter,value=float(parameter_df), min_value=0.0, max_value=8.0, step=0.1)
 parameter_input_values.append(values)
 
input_variables=pd.DataFrame([parameter_input_values],columns=parameter_list,dtype=float)
st.write('\n\n')


if st.button("Click Here to Classify"):
    prediction = model.predict(input_variables)
    if prediction == 0:
        st.image(setosa)
        st.title("setosa")
    elif prediction == 1:
        st.image(versicolor)
        st.title("versiclor")
    else:
        st.image(virginica)
        st.title("virginica")
        