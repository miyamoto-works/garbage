#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from pandas.plotting import scatter_matrix


# In[2]:


#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=Nne)
df = pd.read_csv('winequality-white.csv', sep=';')


# In[3]:


# 機械学習のモデルを作成するトレーニング用と評価用の2種類に分割する
train_x = df.drop(['quality'], axis=1) # 説明変数のみにする
train_y = df['quality'] # 正解クラス
(train_x, test_x ,train_y, test_y) = train_test_split(train_x, train_y, test_size = 0.3, random_state = 42)
#訓練用の説明変数と正解クラス、評価用の説明変数と正解クラスに分割 


# In[4]:


# 識別モデルの構築
random_forest = RandomForestClassifier(max_depth=30, n_estimators=30, random_state=42)
random_forest.fit(train_x, train_y)

# 予測値算出
y_pred = random_forest.predict(test_x)

#モデルを作成する段階でのモデルの識別精度
trainaccuracy_random_forest = random_forest.score(train_x, train_y)
print('TrainAccuracy: {}'.format(trainaccuracy_random_forest))

#作成したモデルに学習に使用していない評価用のデータセットを入力し精度を確認
accuracy_random_forest = accuracy_score(test_y, y_pred)
print('Accuracy: {}'.format(accuracy_random_forest))


# In[5]:


from sklearn.metrics import classification_report
print(classification_report(test_y, y_pred))


# In[6]:


#confusion matrix
mat = confusion_matrix(test_y, y_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False, fmt='d', cmap='RdPu')
plt.xlabel('predicted class')
plt.ylabel('true value')


# In[8]:


# ランダムフォレストのパラメータの候補をいくつか決める
parameters = {
    'n_estimators' :[3,5,10,30,50],#作成する決定木の数
    'random_state' :[7,42],
    'max_depth' :[3,5,8,10],#決定木の深さ
    'min_samples_leaf': [2,5,10,20,50],#分岐し終わったノードの最小サンプル数
    'min_samples_split': [2,5,10,20,50]#決定木が分岐する際に必要なサンプル数
}

#グリッドサーチを使う
from sklearn.model_selection import GridSearchCV
#clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=2, iid=False)
clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=2)

#学習モデルを作成
clf.fit(train_x, train_y)


# In[9]:


#精度を確認
best_clf = clf.best_estimator_ #ここにベストパラメータの組み合わせが入っています
print('score: {:.2%}'.format(best_clf.score(train_x, train_y)))
y_pred = clf.predict(test_x)
print('score: {:.2%}'.format(best_clf.score(test_x, test_y)))


# In[11]:


best_clf


# In[10]:


# 変数の重要度を可視化
importance = pd.DataFrame({ '変数' :train_x.columns, '重要度' :random_forest.feature_importances_})
importance


# In[12]:


plt.scatter(df['Color intensity'], df['Flavanoids'], c = df['Class'])
plt.ylabel('Flavanoids')
plt.xlabel('Color intensity')
plt.show()


# In[ ]:


df_1 = df[df.Class == 1]
df_2 = df[df.Class == 2]
df_3 = df[df.Class == 3]


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("Flavanoids")
ax.set_ylabel("Color intensity")
ax.set_zlabel("Alcohol")
ax.plot(df_1['Flavanoids'], df_1['Color intensity'],df_1['Alcohol'],marker="o",linestyle='None', c='red')
ax.plot(df_2['Flavanoids'], df_2['Color intensity'],df_2['Alcohol'],marker="o",linestyle='None', c='blue')
ax.plot(df_3['Flavanoids'], df_3['Color intensity'],df_3['Alcohol'],marker="o",linestyle='None', c='green')
plt.show()


# In[ ]:


from sklearn import tree
for i in range(1, 31):
    tree.export_graphviz(random_forest.estimators_[i-1],'tree' + str(i) + '.dot')


# In[ ]:




