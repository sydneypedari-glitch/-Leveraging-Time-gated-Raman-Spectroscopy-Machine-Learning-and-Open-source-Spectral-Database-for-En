import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report,  roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold, StratifiedKFold, GridSearchCV, RepeatedKFold
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score


def compute_multiclass_specificity(y_true, y_pred, labels):
    from sklearn.metrics import confusion_matrix
    import numpy as np

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificity_scores = {}

    for idx, label in enumerate(labels):
        TP = cm[idx, idx]
        FP = cm[:, idx].sum() - TP
        FN = cm[idx, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        specificity_scores[label] = specificity
    return specificity_scores

def cross_validation(knn, X, y, number_of_folds):
    k_fold = KFold(n_splits=number_of_folds, shuffle=True, random_state=42)
    y_cv = cross_val_predict(knn, X, y, cv=k_fold, n_jobs=-1)

    target_names = np.unique(y).astype(str)

    print("Classification report:\n", classification_report(y, y_cv, target_names=target_names))
    print("Plotting confusion matrix...\n")

    cm = confusion_matrix(y, y_cv, labels=target_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot()
    plt.show()

    Accuracy = accuracy_score(y, y_cv)
    Precision = precision_score(y, y_cv, average='weighted', zero_division=0)
    Sensitivity_recall = recall_score(y, y_cv, average='weighted', zero_division=0)
    F1_score = f1_score(y, y_cv, average='weighted', zero_division=0)
    Specificity = compute_multiclass_specificity(y, y_cv, labels=target_names)

    
    print("\nMetrics Summary:")
    print({
        "Accuracy": Accuracy,
        "Precision": Precision,
        "Sensitivity_recall": Sensitivity_recall,
        "Specificity": Specificity,
        "F1_score": F1_score
    })


def test_prediction(knn, testing_data_file): 
    test_data = pd.read_csv(testing_data_file)
    test_data.rename(columns={'Unnamed: 0':'Class'}, inplace=True )

    x_test = test_data.iloc[:,1:].values
    y_test = test_data.iloc[:,0].values

    predicted_y = knn.predict(x_test)

    print("Classification report:\n", classification_report(y_test, predicted_y, target_names=knn.classes_))
    cm = confusion_matrix(y_test, predicted_y, labels=knn.classes_)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=knn.classes_)
    disp.plot()
    plt.show()
    import os
    
    results_df = pd.DataFrame({
        'Sample_Index': test_data.index,
        'True_Class': y_test,
        'Predicted_Class': predicted_y
    })

    print("\nPredictions table (first 10 rows):")
    print(results_df.head)

    output_path = os.path.join(os.path.dirname(testing_data_file), 'predicted_classification_table_KNN_all.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nFull classification table saved to: {output_path}")


    Accuracy = accuracy_score(y_test, predicted_y)
    Precision = precision_score(y_test, predicted_y, average='weighted', zero_division=0)
    Sensitivity_recall = recall_score(y_test, predicted_y, average='weighted', zero_division=0)
    F1_score = f1_score(y_test, predicted_y, average='weighted', zero_division=0)
    Specificity = compute_multiclass_specificity(y_test, predicted_y,  labels=knn.classes_)


    print("\nMetrics Summary:")
    print({
        "Accuracy": Accuracy,
        "Precision": Precision,
        "Sensitivity_recall": Sensitivity_recall,
        "specificity" : Specificity, 
        "F1_score": F1_score   })



knn = KNeighborsClassifier(n_neighbors=5)

training_data_file =  #training data file
training_data = pd.read_csv(training_data_file)
training_data.rename(columns={'Unnamed: 0':'Class'}, inplace=True )


X = training_data.iloc[:,1:].values
y = training_data.iloc[:,0].values

knn.fit(X,y)

testing_data_file = #testing data file
cross_validation(knn, X, y, number_of_folds=4)

test_prediction(knn, testing_data_file)