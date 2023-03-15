#scikit-learn is machine learning simple and efficient tool for data mining and data analysis 
#classification is identifying wich catefory an object belongs like spam detection 
#regression is predicting an attribute associated with an object like stock prices predictions 
#clustering i automatic grouping of similar objects into sets like customer segmentation 
#model selection is comparing, validating and choosing parameters and models like imporving model accuracy via parameter tuning 
from statistics import mean
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn import svm 
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
df=pd.read_csv("META.csv") #loading data 
print(df.head())
print(df.info())
print(df.isnull().sum())

#preprocessing data 
a=np.min(df["Low"])
b=np.max(df["Low"])
c=(a+b)/2
bins=(a,c,b)
group_names=["Bad","Good"]
df["Return"]=pd.cut(df["Low"], bins=bins,labels=group_names)
print(df["Return"].unique())
label_return = LabelEncoder() # mean bad = 0 and good =1
df["Return"] = label_return.fit_transform(df["Return"]) # transform data to one command 
print(df.head())
print(df["Return"].value_counts())
sns.countplot(df["Return"])
plt.show()
#seperate our data
X=df.drop("Date", axis=1)
Y=df["Return"]
#train and test splitting of data 
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, random_state=42)
#applying standrd scaling to get optimized result
sc= StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
print(X_train[:4])
# model 1 : random forest classifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,Y_train)
pred_rfc = rfc.predict(X_test)
print(pred_rfc[:5]) #0 are bad and 1 are good 
#perfomance of model 
print(classification_report(Y_test, pred_rfc))
print(confusion_matrix(Y_test,pred_rfc))
#model 2 : SVM classifier 
clf= svm.SVC()
clf.fit(X_train,Y_train)
pred_clf = clf.predict(X_test)
#perfomance of model 
print(classification_report(Y_test, pred_clf))
print(confusion_matrix(Y_test,pred_clf))
#model 3 : neural network 
mlpc=MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=100)
mlpc.fit(X_train,Y_train)
pred_mlpc = mlpc.predict(X_test)
#perfomance of model 
print(classification_report(Y_test, pred_mlpc))
print(confusion_matrix(Y_test,pred_mlpc))
#accuracy score
cm=accuracy_score(Y_test, pred_rfc)
print(cm)