# IMPORTING REQUIRED BUILTIN FUNCTIONS

import pandas as pd
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# IMPORTING DATASET


dataset =  pd.read_csv("https://raw.githubusercontent.com/himanshujha411/LOAN-SANCTION-ASSESSMENT-AND-FEASIBLE-SYSTEM/main/train_data.csv")

new_data =  pd.read_csv("https://raw.githubusercontent.com/himanshujha411/LOAN-SANCTION-ASSESSMENT-AND-FEASIBLE-SYSTEM/main/test_data.csv")

print(dataset.head(20))

# CONVERTING THE STRING INPUTS IN NUMERIC FORM AS THE INDEPENDENT AND DEPENDENT VARIABLES REQUIRES TO BE BINARY

var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    dataset[i] = le.fit_transform(dataset[i])
    
# CREATING THE ARRAY OF TRAIN DATASET AND ACCESSING ITS VALUES
# DIVIDING THE DATASET INTO TRAIN AND TEST DATA
# ONLY CONSIDERING SOME VARIABLES AS ONLY THE MEANINGFUL VARIABLES SHOULD BE INCLUDED
    
array = dataset.values
X = array[:,6:11]
Y = array[:,12]

# CONVERTING TO INTEGER EXPLICITLY BECAUSE ITS SHOWING ERRORS

X = X.astype('int')
Y = Y.astype('int')

#SPLITTING THE SAMPLE DATA :- 80% AS THE TRAINING DATA AND 20% DATA TO BE TESTED

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=10)

# USING LOGISTIC REGRESSION TO PREDICT LOAN APPROVAL
# THIS REQUIRES A LARGE SAMPLE SIZE

model1 = LogisticRegression()           # CREATING OBJECT AND CALLING CONSTRUCTOR
model1.fit(x_train,y_train)             # CALLING THE FIT FUNCTIONS TO TRAIN THE MODEL
predictions = model1.predict(x_test)

print('ACCURACY PERCENTAGE FOR LOGISTIC REGRESSION IS : ',(accuracy_score(y_test, predictions))*100,end='%\n\n')

# USING DECISION TREE TO PREDICT LOAN APPROVAL

model2 = DecisionTreeClassifier()
model2.fit(x_train,y_train)
predictions = model2.predict(x_test)

print('ACCURACY PERCENTAGE FOR DECISION TREE IS : ',(accuracy_score(y_test, predictions))*100,end='%\n\n')

# USING RANDOM FOREST TO PREDICT LOAN APPROVAL
# USING 100 RANDOM RECORDS DATASET 

model3 = RandomForestClassifier(n_estimators=100)
model3.fit(x_train,y_train)
predictions = model3.predict(x_test)

print('ACCURACY PERCENTAGE FOR RONDOM FOREST IS : ',(accuracy_score(y_test, predictions))*100,end='%\n\n')


# USING MODEL TO PREDICT LOAN  APPROVAL AS PER THE MOST ACCURATE RESULT
# CREATING THE ARRAY OF TEST DATASET AND ACCESSING ITS VALUES
# ACCESSING COLUMNS OF THE TEST DATASET TO PREDICT RESULT

array1 = new_data.values
x_test1 = array1[:,6:11]
prediction=model3.predict(x_test1)
print(prediction,'\n')
print('THE ACCURACY SCORE BETWEEN 2 DATASETS ARE : ',(accuracy_score(Y, prediction))*100,end='%')
print('\n\n')


#prediction=str(prediction)
    
#for i in range(len(prediction)):
#    if(prediction[i]=='Y'):
 #       prediction[i] = prediction[i].replace('Y','APPROVED')
  #  elif(prediction[i]=='N'):
   #     prediction[i] = prediction[i].replace('N','NOT APPROVED')
   
   
# CONVERTING THE 1 AND 0 RESULTS INTO 'Y' AND 'N'
   
prediction=le.inverse_transform(prediction)
    
# ADDING NEW COLUMN INTO TEST DATASET AND 

new_data['Loan_Status']=prediction

new_data.to_csv('Loan Approval statement.csv',columns=['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status'])