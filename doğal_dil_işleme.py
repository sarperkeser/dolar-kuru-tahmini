import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

tweetler=pd.read_excel('topluveri.xlsx')



from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

from nltk.corpus import stopwords


derlem =[]
for i in range(285):
    tweet=re.sub('[^a-zA-ZİçÇğüÜıöÖşŞ]',' ',tweetler['tweetler'][i])
    tweet= tweet.lower()
    tweet=tweet.split()
    tweet=[ps.stem(kelime) for kelime in tweet if not kelime in set(stopwords.words('turkish'))]
    tweet= ' '.join(tweet)
    derlem.append(tweet)



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(derlem).toarray() # bağımsız değişken
y = tweetler.iloc[:,3].values # bağımlı değişken

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train = sc.fit_transform(X_train)
x_test = sc.transform(X_test)


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred1 = logr.predict(X_test)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred1)
print("********************************************************************************")
print("Logistic")
#%73
print(cm)



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10, metric='minkowski')
knn.fit(X_train,y_train)

y_pred2 = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred2)
print("********************************************************************************")
print("KNN")
#%58
print(cm)



from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred3 = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred3)
print("********************************************************************************")
print('SVC')
#%69
print(cm)

from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train,y_train)

y_pred3 = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred3)
print("********************************************************************************")
print('SVC linear')
#%71
print(cm)

from sklearn.svm import SVC
svc = SVC(kernel='poly')
svc.fit(X_train,y_train)

y_pred3 = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred3)
print("********************************************************************************")
print('SVC poly')
#%54
print(cm)

from sklearn.svm import SVC
svc = SVC(kernel='sigmoid')
svc.fit(X_train,y_train)

y_pred3 = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred3)
print("********************************************************************************")
print('SVC sigmoid')
#%76
print(cm)



from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred4 = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred4)
print("********************************************************************************")
print('GNB')
#%71
print(cm)


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred5 = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred5)
print("********************************************************************************")
print('DTC')
#%66
print(cm)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'gini')

dtc.fit(X_train,y_train)
y_pred5 = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred5)
print("********************************************************************************")
print('DTC gini')
#%65
print(cm)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred6 = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred6)
print("********************************************************************************")
print('RFC')
#%69
print(cm)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, criterion = 'gini')
rfc.fit(X_train,y_train)

y_pred6 = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred6)
print("********************************************************************************")
print('RFC gini')
#%71
print(cm)

