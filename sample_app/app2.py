# -*- coding: utf-8 -*-
"""
Sample streamlit file
"""
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

@st.cache
def load_data(number_input):
    df = pd.read_csv('https://raw.githubusercontent.com/JonathanBechtel/dat-02-22/main/ClassMaterial/Unit3/data/ks2.csv', nrows=number_input)
    return df
    
@st.cache
def group_data(x_axis, y_axis):
    result = df.groupby(x_axis)[y_axis].mean()
    return result

#@st.cache
def load_model():
    with open('mod.pkl', 'rb') as mod:
        pipe = pickle.load(mod)
        
    return pipe
    
    
section = st.sidebar.radio("App Section", ['Data Explorer','Model Explorer'])
num_rows = st.sidebar.number_input('Number of Rows to laod', min_value=100, value=1000, step=1000)

df = load_data(num_rows)

if section == 'Data Explorer':
    
    st.header("Exploring Kickstart Campaigns!")

    st.write(df)

    chart_type = st.sidebar.selectbox('Chart Type', ['Bar','Line','Strip'])
    x_axis = st.sidebar.selectbox('X Axis Column', ['category','main_category','country'])
    y_axis = st.sidebar.selectbox('Y Axis Column', ['state', 'goal'])


    if chart_type == 'Bar':
        result = group_data(x_axis, y_axis)
        st.bar_chart(result)
    elif chart_type == 'Line':
        result = group_data(x_axis, y_axis)
        st.line_chart(result)
    else:
        result = df[[x_axis, y_axis]]
        st.plotly_chart(px.strip(result, x=x_axis, y=y_axis, color=x_axis))

elif section == 'Model Explorer':

    st.header('Make predictions with your Model')

    pipe = load_model()
    #print(pipe)

    category = st.sidebar.selectbox('Category', df['category'].unique().tolist())
    main_category = st.sidebar.selectbox('Main Category', df['main_category'].unique().tolist())
    goal = st.sidebar.number_input('Fundraising Amount', min_value=0, value=1000, step=500) 
    
    sample = pd.DataFrame({
        'category': [category],
        'main_category': [main_category],
        'funding_amount': [goal]
        })
    
    prediction = pipe.predict_proba(sample)
    #print(prediction)
    positive_prob = prediction[0][1]
    
    st.title(f"Odds of a Successful Campaign Are: {positive_prob:.2%}")
    
    #st.header(f"Predicted Probability of Campaign Successs: {prediction[0][1]:.2%}")