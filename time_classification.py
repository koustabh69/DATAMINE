# Import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

import pandas as pd

pls=pd.read_csv("plane_crash.csv")

pl=pls.head(150)
pl["Time"].fillna("00:00:00", inplace = True)
pl["Operator"].fillna("LOCAL", inplace = True)
pl["Aboard"].fillna("0", inplace = True)

time=pl['Time']

operate=pl['Operator']

aboard=pl['Aboard']

print(aboard.describe())

df.replace("?","") or np.NAN)

#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
operate_encoded=le.fit_transform(operate)
print(operate_encoded)


# Converting string labels into numbers
time_encoded=le.fit_transform(time)
#label=le.fit_transform(play)
print ("Time:",time_encoded)
#print ("Play:",label)

#Combinig weather and temp into single listof tuples
features=tuple(zip(operate_encoded,time_encoded))
print(features)
max_scaler=MaxAbsScaler()
ytrain=aboard
xtrain=max_scaler.fit_transform(features)
print(xtrain)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(xtrain, ytrain, test_size = .1,random_state=1)


#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(X_train,y_train)

#Predict Output
predicted= model.predict(X_train) # 0:Overcast, 2:Mild
print("Predicted Value:", predicted)


#df.values.tolist()   df to list