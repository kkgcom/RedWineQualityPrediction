import numpy as np
import pandas as pd
import matplotlib.pyplot as py
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

df_wine=pd.read_csv('C:\DataScience\Machine Learning\Iinear_regression\QualityPrediction.csv')
df_wine
# print(str(df_wine.shape))
# df_wine['quality'].fillna(0,inplace=True)
badindex=df_wine.loc[df_wine['quality'] <= 5].index
goodindex=df_wine.loc[df_wine['quality'] > 5].index
df_wine.iloc[badindex, df_wine.columns.get_loc('quality')]=0
df_wine.iloc[goodindex, df_wine.columns.get_loc('quality')]=1
x=df_wine.drop('quality',axis=1)
y=df_wine['quality']
# Splitting Training and Test Set
# choosing 20% as  training data
xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.20,random_state=42)
regressor=LinearRegression()

# Fitting model with training data
regressor.fit(xTrain,yTrain)
# y_prediction_lr=regressor.predict(xTest)

# Saving model to disk
pickle.dump(regressor,open('model.pkl','wb'))

# Loading model to compare the results
model=pickle.load(open('model.pkl','rb'))

# print(model.predict([[7.9,0.35,0.46,3.6,0.078,15,37,0.9973,3.35,0.86,12.8]]))
print(model.predict([[7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]]))