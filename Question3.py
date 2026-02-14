import pandas
from sklearn.model_selection import train_test_split


#open CSV and convert to DF and fill in NaN values
kidneyDiseaseDF = pandas.read_csv('kidney_disease.csv',encoding='utf-8-sig', header = 0)
kidneyDiseaseDFnoNaN = kidneyDiseaseDF.replace(r'\s+', '', regex=True).replace(r'\?', 0, regex=True).fillna(0)

# Create a feature matrix X that contains all columns except classification. Create a label vector y using the classification column. 
X = kidneyDiseaseDFnoNaN.drop('classification', axis = 1)
y = kidneyDiseaseDFnoNaN['classification']

#Split the data set 70:30 training set to testing set. Added randomized seed to make repeated runs the same.
features_train, features_test, labels_train, labels_test = train_test_split(X, y,test_size=0.30,random_state=42)

'''
• Why we should not train and test a model on the same data
    - Models should not be trained and tested on the exact dataset as they can memorize the dataset eventually, 
      becoming overfitted, and the "expected" accuracy/precision can be inconsistent when tested on a different dataset.

• What the purpose of the testing set is
    - Testing set provides a dataset with known values that the model wasnt trained on
      to compare the predicted values to. This allows us to verify the model's accuracy and precision. 
'''