# STUDENT-DATA


 Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline
In [79]:
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)
Data imported successfully
Out[79]:
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
5	1.5	20
6	9.2	88
7	5.5	60
8	8.3	81
9	2.7	25
In [80]:
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

In [81]:
X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values
In [82]:
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)
In [83]:
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training completed.")
Training completed.
In [84]:
line = regressor.coef_*X+regressor.intercept_

plt.scatter(X, y)
plt.plot(X, line);
plt.show()

In [85]:
line = regressor.coef_*X+regressor.intercept_

plt.scatter(X, y)
plt.plot(y, line);
plt.show()

In [86]:
print(X_test) 
y_pred = regressor.predict(X_test)
[[1.5]
 [3.2]
 [7.4]
 [2.5]
 [5.9]]
In [87]:
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df
Out[87]:
Actual	Predicted
0	20	16.884145
1	27	33.732261
2	69	75.357018
3	30	26.794801
4	62	60.491033
In [88]:
print(X_test)
y_pred = regressor.predict([[9.25]])
print(y_pred)
[[1.5]
 [3.2]
 [7.4]
 [2.5]
 [5.9]]
[93.69173249]
In [ ]:
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))
Mean Absolute Error: 4.183859899002982
AUTHOR :SAKTHIRAAMBALASUNDARAM
