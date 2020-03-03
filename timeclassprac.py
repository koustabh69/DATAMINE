import numpy as np
from sklearn import preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#importing dataset
#data = pd.read_csv('Absenteeism_at_work1.csv',delimiter=';')
data=pd.read_csv('plane_crash.csv')
#print(data.head())

#preprocessing
data = data.replace("?",np.NAN)
data.fillna(0,inplace = True)
data = data.replace("",np.NAN)


data['Time']=pd.to_datetime(data.Time)
hour=(data.Time.dt.hour)
print(hour.head())

'''
time = data['Time'].str.split(':')
print("vdugis")
print(time)

'''
location=data['Location']


le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
'''
location_encoded=le.fit_transform(location)
print(location_encoded)
'''

#x = data.drop('Month of absence', axis=1)
#x=data[['Time','Location']]
#Combinig weather and temp into single listof tuples
x=tuple(zip(time,location))
print(x)
#print(data.head())

y = data['Operator']
#print(y.head())

from sklearn.model_selection import train_test_split

#Spliting dataset into train and test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=100)
#1xtest.fillna(xtest.mean())

def naive_based():
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()
    nb.fit(xtrain, ytrain)
    ypred = nb.predict(xtest)
    #print(ypred)
    from sklearn import metrics
    print("Accuracy:",metrics.accuracy_score(ytest, ypred))
    print(metrics.classification_report(ytest,ypred))
    print(metrics.confusion_matrix(ytest,ypred))
    sns.heatmap(metrics.confusion_matrix(ytest,ypred))
    plt.show()
   
def decision_tree():
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    classifier.fit(xtrain, ytrain)
    ypred = classifier.predict(xtest)
    #print(ypred)
    from sklearn import metrics
    print("Accuracy:",metrics.accuracy_score(ytest, ypred))
    print(metrics.classification_report(ytest,ypred))
    print(metrics.confusion_matrix(ytest,ypred))
    sns.heatmap(metrics.confusion_matrix(ytest,ypred))
    plt.show()
   
   
print("1. naive based\n2. decision tree")
n = int(input("Enter your choice:"))

if(n == 1):
    naive_based()
else:
    decision_tree()


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(ytest, ypred))
print(metrics.classification_report(ytest,ypred))
print(metrics.confusion_matrix(ytest,ypred))
sns.heatmap(metrics.confusion_matrix(ytest,ypred))
plt.show()
