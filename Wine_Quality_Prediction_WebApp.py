# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:17:04 2024

@author: prachet
"""

import numpy as np
import pickle
import streamlit as st

#loading. the saved model
loaded_model = pickle.load(open('winequality_trained_model.sav','rb'))

#creating a function for prediction

def wine_quality_prediction(input_data):

    #changing the input data to numpy
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting on 1 instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    #print(prediction)

    if(prediction[0]==1):
      return 'Good Quality Wine' 
    else:
      return 'Bad Quality Wine'
  
    
  
def main():

    
    #giving a title
    st.title('Wine Quality Prediction Web App')
    
    #getting input data from user
    
    col1 , col2 = st.columns(2)
    #getting input data from user
    with col1:
        fixed_acidity = st.text_input("Fixed Acidity of Wine")
    with col2:
        volatile_acidity = st.text_input("Volatile Acidity of Wine")
    with col1:
        citric_acid = st.text_input("Citric Acid in Wine")
    with col2:
        residual_sugar = st.text_input("Residual Sugar in Wine")
    with col1:
        chlorides = st.text_input("Chlorides in Wine")
    with col2:
        free_sulfur_dioxide = st.text_input("Free Sulfur-dioxide in Wine")
    with col1:
        total_sulfur_dioxide = st.text_input("Total Sulfur-dioxide in Wine")
    with col2:
        density = st.text_input("Density of Wine")
    with col1:
        pH = st.text_input("pH of Wine")
    with col2:
        sulphates = st.text_input("Sulphates in Wine")
    with col1:
        alcohol = st.text_input("Alcohol in Wine")
   
    # code for prediction
    wine_quality = ''
   
    #creating a button for Prediction
    if st.button('Predict Wine Quality'):
       wine_quality =wine_quality_prediction((fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol)) 
    st.success(wine_quality)
    
    
if __name__ == '__main__':
    main()
    
    
    
