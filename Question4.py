from sklearn.neighbors import KNeighborsClassifier
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score

#open CSV and convert to DF and fill in NaN values
kidneyDiseaseDF = pandas.read_csv('kidney_disease.csv',encoding='utf-8-sig', header = 0)
kidneyDiseaseDFnoNaN = kidneyDiseaseDF.replace(r'\s+', '', regex=True).replace(r'\?', 0, regex=True).fillna(0)

#Turn categorical data to numerical 
categoricalToNumerical = {'no': 0, 'yes': 1, 'good': 1, 'poor':0 , 0:0, 'present':1, 'notpresent':0, 'normal':0, 'abnormal':1 }
categoricalDataCols = ['pc','pcc','ba','htn','rbc','dm','cad','appet','pe','ane']

for col in categoricalDataCols:
    kidneyDiseaseDFnoNaN[col] = kidneyDiseaseDFnoNaN[col].map(categoricalToNumerical)

#Code from Q3
X = kidneyDiseaseDFnoNaN.drop('classification', axis = 1)
y = kidneyDiseaseDFnoNaN['classification']
features_train, features_test, labels_train, labels_test = train_test_split(X, y,test_size=0.30,random_state=42)

#train KNN model
knn_model=KNeighborsClassifier(n_neighbors=5)
trained_knn_model = knn_model.fit(X,y)
predictedTestData = trained_knn_model.predict(features_test)

#Print out
print('Confusion Matrix\n',confusion_matrix(labels_test,predictedTestData))
print(f'Accuracy Score: {accuracy_score(labels_test,predictedTestData):.04f}')
print(f'Precision Score: {precision_score(labels_test,predictedTestData,pos_label="ckd"):.04f}')
print(f'Recall Score: {recall_score(labels_test,predictedTestData,pos_label="ckd"):.04f}')
print(f'F1 Score: {f1_score(labels_test,predictedTestData,pos_label="ckd"):.04f}')


'''
• What True Positive, True Negative, False Positive, and False Negative mean in the context
of kidney disease prediction
    - True Positive: The model acurately predicted a case with kidney disease.
    - True Negative: The model acurately predicted a case without kidney disease.
    - False Positive: The model falsely predicted a case with kidney disease.
    - False Negative: The model falsely predicted a case without kidney disease.

• Why accuracy alone may not be enough to evaluate a classification model
    - Accuracy measures the corectness of the model's ability to guess the predicted value but TP, FP, TN, FN are all
      equally "weighed" so we cant see if the model is better or worse at predicting falses or positives. 

• Which metric is most important if missing a kidney disease case is very serious, and why
    - Since missing a case is very serious and can lead to severe consequences, the more important metric 
      for this model would be the precision score as it measures the model's ability in predicting TP by 
      comparing it to cases that it predicted an FN.

'''