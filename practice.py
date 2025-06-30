import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.tree import random
df=pd.read_csv("C:/Users/chsai/OneDrive/Desktop/titanic competetion/train.csv")
df=df.drop('Cabin',axis=1)
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

##dropping unnecessary columns like name,ticket,id
df=df.drop(['PassengerId','Name','Ticket'],axis=1)


continous_columns=df.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_columns=df.select_dtypes(include='object').columns.tolist()
#print(df[continous_columns].corr())
continous_columns.remove('Survived')

#scaling
scaler=StandardScaler()
scalable=['Age','Fare']
df[scalable]=scaler.fit_transform(df[scalable])

#encoding
encoder=OneHotEncoder(sparse_output=False,drop='first')
encoding_columns=['Sex','Embarked']
encoded_features=encoder.fit_transform(df[encoding_columns])
encoded_df=pd.DataFrame(encoded_features,columns=encoder.get_feature_names_out(encoding_columns),index=df.index)
df=pd.concat([df.drop(encoding_columns,axis=1),encoded_df],axis=1)
#print(df.sample(10))

#training the model
X=df.drop('Survived',axis=1)
Y=df['Survived']
model=LogisticRegression()
model.fit(X,Y)

##working with the test data set
test_df=pd.read_csv("C:/Users/chsai/OneDrive/Desktop/titanic competetion/test.csv")


test_df=test_df.drop('Cabin',axis=1)
test_df['Age']=test_df['Age'].fillna(test_df['Age'].median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
test_df['Embarked']=test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])
test_df=test_df.drop(['PassengerId','Name','Ticket'],axis=1)
#print(test_df.info())

scalable=['Age','Fare']
test_df[scalable]=scaler.transform(test_df[scalable])


test_encoded_features=encoder.transform(test_df[encoding_columns])
test_encoded_df=pd.DataFrame(test_encoded_features,columns=encoder.get_feature_names_out(encoding_columns),index=test_df.index)
test_df=pd.concat([test_df.drop(encoding_columns,axis=1),test_encoded_df],axis=1)

y_pred=model.predict(test_df)
original_test = pd.read_csv("C:/Users/chsai/OneDrive/Desktop/titanic competetion/test.csv")

submission = pd.DataFrame({
    'PassengerId': original_test['PassengerId'],
    'Survived': y_pred.astype(int)
})

submission.to_csv("C:/Users/chsai/OneDrive/Desktop/titanic competetion/submission.csv", index=False)
