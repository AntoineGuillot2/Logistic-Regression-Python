# -*- coding: utf-8 -*-
"""
Logistic regression with python
Prediction of diabetes outcome
@author: Antoi
"""
import pandas as pd
import numpy as np
Diabetes=pd.read_csv('diabetes.csv')
table1=np.mean(Diabetes,0)
table2=np.std(Diabetes,0)

inputData=Diabetes.iloc[:,:8]
outputData=Diabetes.iloc[:,8]


from sklearn.linear_model import LogisticRegression
logit1=LogisticRegression()
logit1.fit(inputData,outputData)

logit1.score(inputData,outputData)




####Model performance
####Classification rate 'by hand'
##Correctly classified
np.mean(logit1.predict(inputData)==outputData)
##True positive
trueInput=Diabetes.ix[Diabetes['Outcome']==1].iloc[:,:8]
trueOutput=Diabetes.ix[Diabetes['Outcome']==1].iloc[:,8]
##True positive rate
np.mean(logit1.predict(trueInput)==trueOutput)
##True negative
falseInput=Diabetes.ix[Diabetes['Outcome']==0].iloc[:,:8]
falseOutput=Diabetes.ix[Diabetes['Outcome']==0].iloc[:,8]
##True negative rate
np.mean(logit1.predict(falseInput)==falseOutput)




###Confusion matrix with sklearn
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
confusion_matrix(logit1.predict(inputData),outputData)

###Roc curve
fpr, tpr,_=roc_curve(logit1.predict(inputData),outputData,drop_intermediate=False)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr, color='red',
         lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlabel('False Positive ')
plt.ylabel('True Positive ')
plt.title('ROC curve')
plt.show()


roc_auc_score(logit1.predict(inputData),outputData)

###Coefficient value
coef_DF=pd.DataFrame(data={'Variable':list(inputData),
'value':(logit1.coef_[0])})

coef_DF_standardised=pd.DataFrame(data={'Variable':list(inputData),
'value':(logit1.coef_[0])*np.std(inputData,axis=0)/np.std(outputData)})

##Real vs predicted plot
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(inputData.iloc[:,1],inputData.iloc[:,5],c=logit1.predict_proba(inputData)[:,1],alpha=0.4)
plt.xlabel('Glucose level ')
plt.ylabel('BMI ')
plt.show()

plt.figure()
plt.scatter(inputData.iloc[:,1],inputData.iloc[:,5],c=outputData,alpha=0.4)
plt.xlabel('Glucose level ')
plt.ylabel('BMI ')
plt.show()

