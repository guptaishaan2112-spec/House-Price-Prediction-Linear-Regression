import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df=pd.read_csv("data1.csv")
a=df.columns
x=df[a[0:-1]]#in sq feet
print(x)
y=df['Price']#in $1000s
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
print('slope(w): ',model.coef_)
print('intercept(b): ' ,model.intercept_)
y_pred=model.predict(x_test)
print(y_pred)
fig, ax = plt.subplots(4,1,figsize=(10,8))
for i,j in zip(['Area','Bedrooms','Floor','Age'],range(x.shape[1])):
    print(i,j)
    ax[j].scatter(df[i],df['Price'],color='orange',alpha=0.5,label='actual data')
    ax[j].scatter(df[i],model.predict(x),color='blue',alpha=0.5,label='predicted data')
fig.legend()
plt.show()
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error 
print(r2_score(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
print(root_mean_squared_error(y_test,y_pred))
