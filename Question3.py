import pandas
from sklearn.model_selection import train_test_split


kidneyDiseaseDF = pandas.read_csv('kidney_disease.csv', encoding="utf-8", header = 0)

target = kidneyDiseaseDF.columns[-1]
xFeatureMatrix = kidneyDiseaseDF.drop(target, axis = 1)
yLabel = kidneyDiseaseDF.drop(xFeatureMatrix, axis = 1)

features_train, features_test, labels_train, labels_test = train_test_split(xFeatureMatrix, yLabel,test_size=0.30,random_state=42)

