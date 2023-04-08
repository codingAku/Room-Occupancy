
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import seaborn as sns
from sklearn.model_selection import train_test_split

#Read the data
df = pd.read_csv('./Occupancy.csv')

#Visualize first 10 rows
print(df.head(10))

#See the shape of the data [row x column]
print(df.shape)

#See the non-null data number and data type
print(df.info())

#reduce correlation
def reduce_correlation(df):
    df2 = df
    df2 = df2[np.abs(df2.Temperature - df2.Temperature.mean()) <= 3*df2.Temperature.std()]
    print("1. Removing the Outliers on 'Temperature' has reduced the data size from {} to {}.".format(len(df), len(df2)))
    print("\n")
    df = df2[np.abs(df2.Light - df2.Light.mean()) <= 3*df2.Light.std()]
    print("2. Removing the Outliers on 'Light' has reduced the data size from {} to {}.".format(len(df2), len(df)))
    print("\n")
    df2 = df[np.abs(df.CO2 - df.CO2.mean()) <= 3*df2.CO2.std()]
    print("3. Removing the Outliers on 'CO2' has reduced the data size from {} to {}.".format(len(df), len(df2)))
    print("\n")
    return df2

#Data visualization
def occupancy_plot(df, cat, ax):
    ax.plot(np.where(df.Occupancy == 1, df[cat], None), label='Occupied')
    ax.plot(np.where(df.Occupancy == 0, df[cat], None), label='Not occupied', ls='--')
    ax.grid()
    ax.legend()


fig, axs = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axs.flatten()):
    if i < 5:
        occupancy_plot(df, df.columns[i+1], ax)
        ax.set_title(df.columns[i+1])
    else:
        ax.set_visible(False)

plt.suptitle('Occupancy by Category')
plt.tight_layout()
plt.show()


#See the correlation
print(df.corr()['Occupancy'])


#Correlation matrix
mask = np.triu(np.ones_like(df.corr()))
plt.figure(figsize = (15,8))
sns.heatmap(df.corr(),annot=True, fmt="1.2f", mask=mask, cmap="YlGnBu")
plt.yticks(rotation=0)
plt.show()

df = reduce_correlation(df)

#Split data
X, Y = df.iloc[:, 1:-1], df.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Create a dataframe that used to store data from confusion matrix and accuracy
result = pd.DataFrame(
    columns=['Classifier', 'True Negative', 'False Postive', 'False Negative', 'True Positive', 'Classifier Accuracy'])


def accuracy_vis(xtest, ytest, ypred, predit_proba, cat):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(cat)

    # Confusion Matrix Visulation
    cm = confusion_matrix(ytest, ypred)
    x_axis_labels = ['Actual Postive', 'Actual Negative']
    y_axis_labels = ['Predicted Postive', 'Predicted Negative']
    sns.heatmap(cm, fmt=".0f", annot=True, linewidths=.5, ax=ax1,
                cmap="YlGnBu", xticklabels=x_axis_labels)
    ax1.set_yticklabels(y_axis_labels, rotation=0, ha='right')

    # ROC Curve Visulation
    logit_roc_auc = roc_auc_score(ytest, ypred)
    fpr, tpr, thresholds = roc_curve(ytest, predit_proba[:, 1])
    ax2.plot(fpr, tpr, label='Logistic Regression (area = {})'.format(round(logit_roc_auc, 6)))
    ax2.plot([0, 1], [0, 1], 'r--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend()
    plt.show()
    return (confusion_matrix(Y_test, ypred).ravel())

#Logistic Regression

lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred, lr_score, predit_proba = lr.predict(X_test), lr.score(X_test, Y_test), lr.predict_proba(X_test)
print('Accuracy of Logistic Regression Classifier on test set: {:.6f}%'.format(lr_score*100))
tn, fp, fn, tp = accuracy_vis(X_test, Y_test, Y_pred, predit_proba, 'Logistic Regression')
result.loc['LR'] = ['Logistic Regression', tn, fp, fn, tp, round(lr_score*100, 6)]

X = df[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
y = df['Occupancy']


#Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
Y_pred, dt_score, predit_proba = dt.predict(X_test), dt.score(X_test, Y_test), dt.predict_proba(X_test)
print('Accuracy of Decision Tree Classifier on test set: {:.6f}%'.format(dt_score*100))
tn, fp, fn, tp = accuracy_vis(X_test, Y_test, Y_pred, predit_proba, 'Decision Tree')
result.loc['DT'] = ['Decision Tree', tn, fp, fn, tp, round(dt_score*100, 6)]

#Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
Y_pred, rf_score, predit_proba = rf.predict(X_test), rf.score(X_test, Y_test), rf.predict_proba(X_test)
print('Accuracy of Random Forest Classifier on test set: {:.6f}%'.format(rf_score*100))
tn, fp, fn, tp = accuracy_vis(X_test, Y_test, Y_pred, predit_proba, 'Random Forest')
result.loc['RF'] = ['Random Forest', tn, fp, fn, tp, round(rf_score*100, 6)]



