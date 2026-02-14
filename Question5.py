from sklearn.neighbors import KNeighborsClassifier
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#open CSV and convert to DF
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

#train 5 KNN models
kValues = [1,3,5,7,9]
accuracyScoreValues = []

for k in kValues:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    trained_knn_model = knn_model.fit(X,y)
    predictedTestData = trained_knn_model.predict(features_test)
    accuracyScoreValues.append(accuracy_score(labels_test,predictedTestData))

KNNmodelAccuracyTableData = {'k-value':kValues,'accuracy_score':accuracyScoreValues}
accuracyScoreDF = pandas.DataFrame(KNNmodelAccuracyTableData)

#prints table of accuracy score for the 5 different k values.
print(accuracyScoreDF)

'''
• How changing k affects the behavior of the model
• Why very small values of k may cause overfitting
• Why very large values of k may cause underfitting

    - As k is the number of neighbouring data points that the model consults. The relative value given to the 
      data point's feature will be based on the distance from its k nearest neighbours. This relative value will 
      be whats used to classify other datasets with same features. Higher k means more data points will be used to 
      create the data point's given value and can lead to underfitting as the model will be more sensitive and will 
      be less inclined to make a prediction that doesnt fit the training set values when used on a data set that isnt
      the training set as the high k value used took away its generalisability. The opposite occurs when k value is
      small. Small K value leads to overfitting as the given range of value given to a data point will be wider 
      leading to the model more prone to classifying a data point wrong as it is not as strict when comparing features.


'''