import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

wine_data_set = pd.read_csv('winequality-red.csv')

X = wine_data_set.iloc[:,:-1]
y = wine_data_set.iloc[:,-1].apply(lambda yval : 1 if yval >= 7 else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=45)

model = RandomForestClassifier(n_estimators=200, random_state=2)
model.fit(X_train, y_train)
pred_y = model.predict(X_test)
print("The Accuracy Score is :", accuracy_score(pred_y, y_test))


# Web app

st.title("Wine quality prediction model")
input_data = st.text_input("Enter all wine features")
input_data_split = input_data.split(',')
features = np.asanyarray(input_data_split).reshape(1,-1)

prediction = model.predict(features)

if prediction[0] == 1:
    st.write("This is a good quality 'WINE'.")
else:
    st.write("This is bad quality 'WINE'.")