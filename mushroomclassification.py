import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import  f1_score

df = pd.read_csv("mushroom_data.csv")
df.head()

df['class'].unique()

"""Burada zehirli ve yenilebilir mantarları ikiye ayırmış olduk. Zehirli mantar: p, yenilebilir olanlar: e"""

df.info()

"""Veri setimizi okuma ve inceleme işlemini yapmış olduk. Şimdi veri setimizde bulunan kategorik verileri sayısal verilere dönüştüreceğiz. Bunun için pandas kütüphanesi dummy variable ve scikit-learn kütüphanesindeki LabelEncoder tekniklerini kullanacağız."""

X = df.drop(['class'],axis=1)
y = df['class']

"""Hedef değişken ve bağımlı değişkenimizi ayarladık. Y değişkenine LabelEncoder tekniği uygulayarak, X değişkenine de get_dummies fonksiyonu uygulayarak kategorik verilerimizi sayısal verilere dönüştüreceğiz.

"""

X = pd.get_dummies(X)
X = X.astype(int)
X.head()

encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y)

"""Zehirli (p): 1
Yenebilir(e): 0

Şimdi veri setimizi eğitim ve test kümelerine ayıracağız. Bunun için scikit-learn kütüphanesinin model_selection modülünden yararlanacağız.
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train.shape , X_test.shape

y_train.shape , y_test.shape

"""Karar Ağacı modelini oluşturalım."""

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=0)
clf_gini.fit(X_train, y_train)

plt.figure(figsize=(12,8))

tree.plot_tree(clf_gini.fit(X_train, y_train))

y_pred_gini = clf_gini.predict(X_test)

y_pred_train_gini = clf_gini.predict(X_train)

y_pred_train_gini

"""Modelin ve eğitim setinin doğruluk puanını belirleyelim."""

print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))

print(classification_report(y_test, y_pred_gini))

cm = confusion_matrix(y_test, y_pred_gini)

print('Confusion matrix\n\n', cm)

f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cm, annot=True, linewidths=0.5,linecolor="red", fmt= '.0f',ax=ax)
plt.show()
plt.savefig('ConfusionMatrix.png')

print(classification_report(y_test, y_pred_gini))

f1_score = f1_score(y_test, y_pred_gini)
print("F1 Score:",f1_score)

"""F1 puanı 0.9987 oldukça yüksek bir değerdir. Bu, modelimizin sınıflandırma görevinde çok iyi performans gösterdiğini ve dengeli bir hassasiyet ve geri çağırma oranına sahip olduğunu gösterir. Yani, modelimizin sınıflandırma tahminlerinin hem doğru hem de eksiksiz olduğunu söyleyebiliriz."""

