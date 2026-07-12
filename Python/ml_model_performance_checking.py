import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score



"""
    This file contains different methods to report the performance of a model.

    https://www.kaggle.com/code/prashant111/svm-classifier-tutorial/notebook
    https://scikit-learn.org/stable/modules/model_evaluation.html      
"""



def check_null_accuracy(y : pd.DataFrame) -> float :
    """
    We must compare model accuracy with the null accuracy. Null accuracy is the accuracy 
    that could be achieved by always predicting the most frequent class.
    """
    print("\n[Null accuracy] \n\t y_test :")
    y_test_distribution = y.value_counts()
    for idx, item in enumerate(y_test_distribution) :
        print(f"\t\t {y_test_distribution.index[idx]} : {item} values")
    min = y_test_distribution.min()
    max = y_test_distribution.max()
    null_accuracy = (max/(min+max))
    print('\t Null accuracy score: {0:0.4f}'. format(null_accuracy))
    return null_accuracy


def check_confusion_matrix(y : pd.DataFrame, y_pred : pd.DataFrame) :
    """
    A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. 
    It gives us a summary of correct and incorrect predictions broken down by each category.
    """
    # build matrix
    cm = confusion_matrix(y, y_pred)
    cm_matrix = pd.DataFrame(
        data=cm,
        columns=['Predicted 0', 'Predicted 1'], 
        index=['True 0', 'True 1']
    )
    # add title and label
    labels =  np.array([[f'TN {cm[0,0]}',f'FP {cm[0,1]}'],[f'FN {cm[1,0]}',f'TP {cm[1,1]}']])
    plt.title(f"Confusion matrix")
    # plot matrix
    sns.heatmap(cm_matrix, annot=labels, fmt='', cmap='YlGnBu')
    # build custom legend
    msg = f"{y.name.replace('_', ' ')} : \n"
    distribution = y.value_counts()
    for idx, item in enumerate(y.value_counts()) :
        msg += f"\n    {distribution.index[idx]} : {item} values ({round(item / len(y) * 100, 2)} %)" 
    plt.legend(
        ['Nothing', 'Nothing again'],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=True,
        ncol=5,
        title=msg
    )
    # show
    plt.show()
    """
    # Minimal other version
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y,
        y_pred=y_pred,
        cmap='YlGnBu'
    )
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()
    """
    

def check_classification_report(y : pd.DataFrame, y_pred : pd.DataFrame, print_bla_bla : bool = False, print_as_text : bool = False) :
    """
    """
    if print_bla_bla :
        print(f"\n[Classification Report]")
        print(classification_report(y, y_pred))
        print("\t Precision can be defined as the percentage of correctly predicted positive outcomes out of all the predicted positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true and false positives (TP + FP).")
        print("\t Recall can be defined as the percentage of correctly predicted positive outcomes out of all the actual positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true positives and false negatives (TP + FN). Recall is also called Sensitivity.")
        print("\t f1-score is the weighted harmonic mean of precision and recall. The best possible f1-score would be 1.0 and the worst would be 0.0. f1-score is the harmonic mean of precision and recall.")
        print("\t Support is the actual number of occurrences of the class in our dataset.")
    if print_as_text :
        print(classification_report(y, y_pred))
    else : 
        sns.heatmap(pd.DataFrame(classification_report(y, y_pred, output_dict=True)).iloc[:-1, :].T, annot=True, cmap='RdYlGn')
        plt.show()


def check_roc_auc(estimator, X_test : pd.DataFrame, y_test : pd.DataFrame):
    """
    ROC Curve & ROC-AUC (Receiver Operating Characteristic Curve - Area Under Curve)
    """
    y_score = estimator.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=estimator.classes_[1])
    roc_auc = auc(fpr, tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="SVM").plot()
    plt.show()


def check_roc_auc_old(y : pd.DataFrame, y_pred : pd.DataFrame):
    """
    ROC Curve & ROC-AUC (Receiver Operating Characteristic Curve - Area Under Curve)
    > The ROC Curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold levels.
    True Positive Rate (TPR) is also called Recall. It is defined as the ratio of TP to (TP + FN).
    False Positive Rate (FPR) is defined as the ratio of FP to (FP + TN).
    > ROC AUC measures the area under the curve
    """
    print("\n [ROC Curve & ROC-AUC (Receiver Operating Characteristic Curve - Area Under Curve)]")
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    ROC_AUC = round(roc_auc_score(y, y_pred), 4)
    plt.figure("ROC curve for Predicting Stress classifier", figsize=(6,4))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1], [0,1], 'k--' )
    plt.rcParams['font.size'] = 12
    plt.title(f'ROC AUC : {ROC_AUC} (1.0=perfect, 0.5=random)')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.show()
