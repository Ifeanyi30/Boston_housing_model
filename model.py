import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

df = pd.read_csv("Boston.csv")

x = df[['crim', 'chas', 'rm']]
y = df[['medv']]

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.72, test_size= 0.28, random_state=101, shuffle=True)
regr = LinearRegression()

regr.fit(x_train,y_train)
regr.score(x_test, y_test)

print(regr.predict([[0.1,0,6]]))