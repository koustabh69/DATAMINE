# Import LabelEncoder
from sklearn import preprocessing
import pandas as pd
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
 
import matplotlib.pyplot as plt
plt.ion()
plt.style.use('ggplot')
fig=plt.figure()
fig,ax=plt.subplots()


pl=pd.read_csv("AirQualityUCI.csv", delimiter=';')
pima =pl.head(100) 

print(pima.head(5))

A=pima['PT08.S4(NO2)']
B=pima['PT08.S5(O3)']

label=pima['PT08.S1(CO)']
data=tuple(zip(A,B))



print(data)


#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(data,label)


pis=pl[100:150]

Anew=pis['PT08.S4(NO2)']
Bnew=pis['PT08.S5(O3)']

datanew=tuple(zip(Anew,Bnew))


#Predict Output

print("predictions ")
predicted= model.predict(datanew) # 0:Overcast, 2:Mild
print("Predicted Value:", predicted)


ts = pd.Series(predicted ,index=pd.date_range('1/1/2000', periods=50))

plt.plot(pd.Series(predicted ,index=pd.date_range('1/1/2000', periods=50))
)

print("REAL VALUES \n")

c=pis['PT08.S1(CO)']

ti = pd.Series(c ,index=pd.date_range('1/1/2000', periods=50))

ti = ti.cumsum()
ax.plot(ti)
plt.show(ax)
plt.show(plt)



