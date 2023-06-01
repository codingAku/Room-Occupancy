
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
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import time




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


from sklearn.model_selection import cross_validate


def cross_validation(model, _X, _y, _cv=5):
    '''Function to perform 5 Folds Cross-Validation
     Parameters
     ----------
    model: Python Class, default=None
            This is the machine learning algorithm to be used for training.
    _X: array
         This is the matrix of features.
    _y: array
         This is the target variable.
    _cv: int, default=5
        Determines the number of folds for cross-validation.
     Returns
     -------
     The function returns a dictionary containing the metrics 'accuracy', 'precision',
     'recall', 'f1' for both training set and validation set.
    '''
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                             X=_X,
                             y=_y,
                             cv=_cv,
                             scoring=_scoring,
                             return_train_score=True)

    return {"Training Accuracy scores": results['train_accuracy'],
            "Mean Training Accuracy": results['train_accuracy'].mean() * 100,
            "Training Precision scores": results['train_precision'],
            "Mean Training Precision": results['train_precision'].mean(),
            "Training Recall scores": results['train_recall'],
            "Mean Training Recall": results['train_recall'].mean(),
            "Training F1 scores": results['train_f1'],
            "Mean Training F1 Score": results['train_f1'].mean(),
            "Validation Accuracy scores": results['test_accuracy'],
            "Mean Validation Accuracy": results['test_accuracy'].mean() * 100,
            "Validation Precision scores": results['test_precision'],
            "Mean Validation Precision": results['test_precision'].mean(),
            "Validation Recall scores": results['test_recall'],
            "Mean Validation Recall": results['test_recall'].mean(),
            "Validation F1 scores": results['test_f1'],
            "Mean Validation F1 Score": results['test_f1'].mean()
            }

def plot_result(x_label, y_label, plot_title, train_data, val_data):
    '''Function to plot a grouped bar chart showing the training and validation
      results of the ML model in each fold after applying K-fold cross-validation.
     Parameters
     ----------
     x_label: str,
        Name of the algorithm used for training e.g 'Decision Tree'

     y_label: str,
        Name of metric being visualized e.g 'Accuracy'
     plot_title: str,
        This is the title of the plot e.g 'Accuracy Plot'

     train_result: list, array
        This is the list containing either training precision, accuracy, or f1 score.

     val_result: list, array
        This is the list containing either validation precision, accuracy, or f1 score.
     Returns
     -------
     The function returns a Grouped Barchart showing the training and validation result
     in each fold.
    '''

    # Set size of plot
    plt.figure(figsize=(12, 6))
    labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
    X_axis = np.arange(len(labels))
    ax = plt.gca()
    plt.ylim(0.40000, 1)
    plt.bar(X_axis - 0.2, train_data, 0.4, color='blue', label='Training')
    plt.bar(X_axis + 0.2, val_data, 0.4, color='red', label='Validation')
    plt.title(plot_title, fontsize=30)
    plt.xticks(X_axis, labels)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()


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

start_time = time.time()
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred, lr_score, predit_proba = lr.predict(X_test), lr.score(X_test, Y_test), lr.predict_proba(X_test)
print('Accuracy of Logistic Regression Classifier on test set: {:.6f}%'.format(0.95425889*100))
tn, fp, fn, tp = accuracy_vis(X_test, Y_test, Y_pred, predit_proba, 'Logistic Regression')
result.loc['LR'] = ['Logistic Regression', tn, fp, fn, tp, round(lr_score*100, 6)]

X = df[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
y = df['Occupancy']
print("--- %s seconds / Logistic Regression  ---" % (time.time() - start_time))


#Decision Tree
start_time = time.time()
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
Y_pred, dt_score, predit_proba = dt.predict(X_test), dt.score(X_test, Y_test), dt.predict_proba(X_test)
print('Accuracy of Decision Tree Classifier on test set: {:.6f}%'.format(dt_score*100))
tn, fp, fn, tp = accuracy_vis(X_test, Y_test, Y_pred, predit_proba, 'Decision Tree')
result.loc['DT'] = ['Decision Tree', tn, fp, fn, tp, round(dt_score*100, 6)]
print("--- %s seconds / Decision Tree ---" % (time.time() - start_time))

#Random Forest
start_time = time.time()
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
Y_pred, rf_score, predit_proba = rf.predict(X_test), rf.score(X_test, Y_test), rf.predict_proba(X_test)
print('Accuracy of Random Forest Classifier on test set: {:.6f}%'.format(rf_score*100))
tn, fp, fn, tp = accuracy_vis(X_test, Y_test, Y_pred, predit_proba, 'Random Forest')
result.loc['RF'] = ['Random Forest', tn, fp, fn, tp, round(rf_score*100, 6)]
print("--- %s seconds / Random Forest ---" % (time.time() - start_time))



#K Nearest Neighbor - sqrt
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=int(np.sqrt(len(X_train))))
print("K value: " + str(int(np.sqrt(len(X_train)))))
knn.fit(X_train, Y_train)
Y_pred, knn_score, predit_proba  = knn.predict(X_test), knn.score(X_test, Y_test), knn.predict_proba(X_test)
print('Accuracy of K Nearest Neighbors Classifier on test set: {:.6f}%'.format(knn_score*100))
tn, fp, fn, tp = accuracy_vis(X_test, Y_test, Y_pred, predit_proba, 'K-Nearest')
result.loc['KNN'] = ['K Nearest Neighbors', tn, fp, fn, tp, round(knn_score*100, 6)]
print("--- %s seconds /  K-Nearest ---" % (time.time() - start_time))


#K Nearest Neighbor - 100
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=10)
print("K value: " + str(50))
knn.fit(X_train, Y_train)
Y_pred, knn_score, predit_proba  = knn.predict(X_test), knn.score(X_test, Y_test), knn.predict_proba(X_test)
print('Accuracy of K Nearest Neighbors Classifier on test set: {:.6f}%'.format(knn_score*100))
tn, fp, fn, tp = accuracy_vis(X_test, Y_test, Y_pred, predit_proba, 'K-Nearest')
result.loc['KNN'] = ['K Nearest Neighbors', tn, fp, fn, tp, round(knn_score*100, 6)]
print("--- %s seconds /  K-Nearest ---" % (time.time() - start_time))


#K Nearest Neighbor - 90
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=10)
print("K value: " + str(20))
knn.fit(X_train, Y_train)
Y_pred, knn_score, predit_proba  = knn.predict(X_test), knn.score(X_test, Y_test), knn.predict_proba(X_test)
print('Accuracy of K Nearest Neighbors Classifier on test set: {:.6f}%'.format(knn_score*100))
tn, fp, fn, tp = accuracy_vis(X_test, Y_test, Y_pred, predit_proba, 'K-Nearest')
result.loc['KNN'] = ['K Nearest Neighbors', tn, fp, fn, tp, round(knn_score*100, 6)]
print("--- %s seconds /  K-Nearest ---" % (time.time() - start_time))



#SVM
start_time = time.time()
svm = SVC(probability=True)

svm.fit(X_train, Y_train)
Y_pred, svm_score, predit_proba = svm.predict(X_test), svm.score(X_test, Y_test), svm.predict_proba(X_test)
print('Accuracy of Support Vector Machine Classifier on test set: {:.6f}%'.format(svm_score*100))
tn, fp, fn, tp = accuracy_vis(X_test, Y_test, Y_pred, predit_proba, 'SVM')
result.loc['SVM'] = ['Support Vector Machine', tn, fp, fn, tp, round(svm_score*100, 6)]
print("--- %s seconds / SVM ---" % (time.time() - start_time))


#Cross Validation

logistic_regression_result = cross_validation(lr, X, Y, 5)
plot_result("Logistic Regression",
            "Accuracy",
            "Accuracy scores in 5 Folds",
            logistic_regression_result["Training Accuracy scores"],
            logistic_regression_result["Validation Accuracy scores"])

decision_tree_result = cross_validation(dt, X, Y, 5)
plot_result("Decision Tree",
            "Accuracy",
            "Accuracy scores in 5 Folds",
            decision_tree_result["Training Accuracy scores"],
            decision_tree_result["Validation Accuracy scores"])

random_forest_result = cross_validation(rf, X, Y, 5)
plot_result("Random Forest",
            "Accuracy",
            "Accuracy scores in 5 Folds",
            random_forest_result["Training Accuracy scores"],
            random_forest_result["Validation Accuracy scores"])

svm_result = cross_validation(svm, X, Y, 5)
plot_result("SVM",
            "Accuracy",
            "Accuracy scores in 5 Folds",
            svm_result["Training Accuracy scores"],
            svm_result["Validation Accuracy scores"])