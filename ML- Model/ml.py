"""Import Packages"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)



"""Import data as data frame"""
df = pd.read_csv('Diabetestype.csv')

df.head()


df.info()


df.describe()


df.drop("Class", 1, inplace=True)
df


"""Exploratory Data Analysis"""
plt.figure(figsize=(12,8))
sns.countplot(df.Type)


plt.figure(figsize=(50,10))
sns.countplot(df.Type, hue=df.Age)


plt.figure(figsize=(20,8))
sns.boxplot(x='Age', y='BS Fast', data=df)
plt.title('Box Plot of Age with Blood Sugar Level')


plt.figure(figsize=(20,8))
sns.scatterplot(x='Age', y='BS Fast', data=df, hue='Type')
plt.title('Box Plot of Age with Blood Sugar Level')


plt.figure(figsize=(18,15))
sns.heatmap(df.corr(), annot = True, cmap = 'RdYlGn')


"""Import Packages"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


"""Building the ML Model"""
X = df.drop('Type', 1)
y = df.iloc[:,-1]

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)

"""Testing different base models"""
logreg = LogisticRegression()
logreg_cv = LogisticRegressionCV()
rf= RandomForestClassifier()
gboost = GradientBoostingClassifier()
models = [logreg, logreg_cv, rf, gboost]

for model in models:
    print('Cross-validation of : {0}'.format(model.__class__))
    score = compute_score(clf=model, X=X, y=y, scoring='accuracy')
    print('CV score = {0}'.format(score))
    print('****')

"""Lets try training test split"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

"""Using the random forest algorithm"""
model = rf.fit(X_train, y_train)

y_pred = model.predict(X_test)

"""Check the prediction precision and accuracy"""
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

"""Saving the model with pickle"""
import pickle

"""Save the model to disk"""
model_name = 'model.pkl'
pickle.dump(model, open(model_name, 'wb'))

print("[INFO]: Finished Saving Model.")
