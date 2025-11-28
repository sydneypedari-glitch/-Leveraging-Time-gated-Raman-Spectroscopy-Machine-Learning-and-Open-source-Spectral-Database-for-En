import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold, StratifiedKFold, GridSearchCV, RepeatedKFold 
from sklearn import metrics

def feature_selection(rf, training_data, number_of_features):
    
    ftName = list(training_data.columns.values) 
    del ftName[0]
   
    for x in range(0, len(ftName)): 
        ftName[x] = float(ftName[x])
    
    importances = rf.feature_importances_ 
    feature_importance_df = pd.DataFrame({'Feature': ftName, 'Importance': importances}) 

    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    top_features = feature_importance_df['Feature'][:number_of_features].values 
    top_features = map(str, top_features) 

    X=training_data.loc[: , top_features].values 
    return(X) 

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

 
def cross_validation(rf, X, y):
    number_of_folds = 4
    k_fold = StratifiedKFold(n_splits=number_of_folds, shuffle=False, random_state=None)
    y_cv = cross_val_predict(rf, X, y, cv=k_fold, n_jobs=-1)

    print("Classification report:\n", classification_report(y, y_cv, target_names=rf.classes_))

    cm = confusion_matrix(y, y_cv, labels=rf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
    disp.plot()
    plt.show()

    Accuracy = accuracy_score(y, y_cv)
    Precision = precision_score(y, y_cv, average='weighted', zero_division=0)
    Sensitivity_recall = recall_score(y, y_cv, average='weighted', zero_division=0)
    F1_score = f1_score(y, y_cv, average='weighted', zero_division=0)
    Specificity = compute_multiclass_specificity(y, y_cv, labels=rf.classes_)

    
    print("\nMetrics Summary:")
    print({
        "Accuracy": Accuracy,
        "Precision": Precision,
        "Sensitivity_recall": Sensitivity_recall,
        "Specificity": Specificity,
        "F1_score": F1_score
    })

def test_prediction(rf, y):
    import os
    testing_data_file = #testing data
    test_data = pd.read_csv(testing_data_file)
    test_data.rename(columns={'Unnamed: 0':'Class'}, inplace=True )

    x_test = test_data.iloc[:,1:].values
    y_test = test_data.iloc[:,0].values 
    unique_count_t = len(set(y_test))
    print(f"The number of unique values is: {unique_count_t}")
   
    predicted_y = rf.predict(x_test) 

    print("Classification report:\n", classification_report(y_test, predicted_y, target_names=rf.classes_))
    cm = confusion_matrix(y_test, predicted_y, labels=rf.classes_) 

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=rf.classes_) 
    disp.plot()
    plt.show()

    results_df = pd.DataFrame({
        'Sample_Index': test_data.index,
        'True_Class': y_test,
        'Predicted_Class': predicted_y
    })

  
    output_path = os.path.join(os.path.dirname(testing_data_file), 'predicted_classification_table_RF_all.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nFull classification table saved to: {output_path}")

    Accuracy = accuracy_score(y_test, predicted_y)
    Precision = precision_score(y_test, predicted_y, average='weighted', zero_division=0)
    Sensitivity_recall = recall_score(y_test, predicted_y, average='weighted', zero_division=0)
    F1_score = f1_score(y_test, predicted_y, average='weighted', zero_division=0)
    Specificity = compute_multiclass_specificity(y_test, predicted_y,  labels=rf.classes_)


    print("\nMetrics Summary:")
    print({
        "Accuracy": Accuracy,
        "Precision": Precision,
        "Sensitivity_recall": Sensitivity_recall,
        "specificity" : Specificity, 
        "F1_score": F1_score   })


def main():
    
    training_data_file =#training data
    training_data = pd.read_csv(training_data_file)
    training_data.rename(columns={'Unnamed: 0':'Class'}, inplace=True )

    X = training_data.iloc[:,1:].values
    y = training_data.iloc[:,0].values
    unique_count = len(set(y))
    print(f"The number of unique values is: {unique_count}")
    rf = RandomForestClassifier(n_estimators=300, bootstrap=True, random_state=13, class_weight='balanced')
    rf.fit(X,y)

    number_of_top_features = input("How many top features do you want to keep? (Leave blank to use all): ") or None
    if number_of_top_features != None:
        number_of_top_features = int(number_of_top_features)
        X = feature_selection(rf,training_data, number_of_top_features)
        rf.fit(X,y)
   
    
    while True:
        print("Select the following options:")
        print(" "*4, "1. Perform cross-validation.")
        print(" "*4, "2. Perform test set prediction.")
        print(" "*4, "3. Exit.")
        option = input("Enter your selection: ")
        if (option == "1"):
            cross_validation(rf, X, y)
        elif option == "2":
            test_prediction(rf, y)
        elif option == "3":
            break



if __name__ == "__main__":
    main()