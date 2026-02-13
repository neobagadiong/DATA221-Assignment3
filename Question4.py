from sklearn.neighbors import KNeighborsClassifier
import pandas
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer



kidneyDiseaseDF = pandas.read_csv('kidney_disease.csv',encoding='utf-8-sig', header = 0)
kidneyDiseaseDF = kidneyDiseaseDF.drop('rbc',axis=1)
kidneyDiseaseDF[['pc','pcc','ba','pcv','wc','rc','htn','dm','cad','appet','pe','ane']].map(lambda x: str(x).strip().encode('utf-8').decode('ascii', 'ignore'))

kidneyDiseaseDFnoNaN = kidneyDiseaseDF.fillna(0)
print(kidneyDiseaseDFnoNaN.groupby('pcv').count())
print(kidneyDiseaseDFnoNaN.groupby('wc').count())
print(kidneyDiseaseDFnoNaN.groupby('rc').count())
print(kidneyDiseaseDFnoNaN.groupby('htn').count())
print(kidneyDiseaseDFnoNaN.groupby('dm').count())
print(kidneyDiseaseDFnoNaN.groupby('cad').count())
print(kidneyDiseaseDFnoNaN.groupby('appet').count())
print(kidneyDiseaseDFnoNaN.groupby('pe').count())
print(kidneyDiseaseDFnoNaN.groupby('ane').count())

'''
pc = abnomal,normal
pcc = present,not present
ba = present,notpresent



'''


'''print(kidneyDiseaseDF[kidneyDiseaseDF.isna().any(axis=1)])

target = kidneyDiseaseDFnoNaN.columns[-1]
xFeatureMatrix = kidneyDiseaseDFnoNaN.drop(target, axis = 1)
yLabel = kidneyDiseaseDFnoNaN.drop(xFeatureMatrix, axis = 1)
features_train, features_test, labels_train, labels_test = train_test_split(xFeatureMatrix, yLabel,test_size=0.30,random_state=42)

knn_model=KNeighborsClassifier(n_neighbors=5)
trained_knn_model = knn_model.fit(xFeatureMatrix,yLabel)
'''

#predictedTestData = trained_knn_model.predict(features_test)