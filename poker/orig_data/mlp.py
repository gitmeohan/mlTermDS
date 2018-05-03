#-------------------------------------------------------------------------
# All the Libraries:
#-------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
#----------------------------------------------------------------
#Read the Training and Testing Data:
#----------------------------------------------------------------
train_data = pd.read_csv(filepath_or_buffer="poker-hand-training-true.data", sep=',', header=None)
test_data = pd.read_csv(filepath_or_buffer="poker-hand-testing.data", sep=',', header=None)
verify_data=pd.read_csv(filepath_or_buffer="poker-hand-testing-personal.data", sep=',', header=None)
#----------------------------------------------------------------
#Print it's Shape to get an idea of the data set:
#----------------------------------------------------------------
print("Train data shape:",train_data.shape)
print("Test date shape:",test_data.shape)
#----------------------------------------------------------------
#Prepare the Data for Training and Testing:
#----------------------------------------------------------------
#Ready the Train Data
train_data_array = train_data.values
train_data = train_data_array[:,0:10]
train_data_label = train_data_array[:,10]
#Ready the Test Data
test_data_array = test_data.values
test_data = test_data_array[:,0:10]
test_data_label = test_data_array[:,10]

verify_data_array=verify_data.values
verify_data=verify_data_array[:,0:10]
verify_data_label=verify_data_array[:,10]

#----------------------------------------------------------------
# Scaling the Data for our Main Model
#----------------------------------------------------------------
# Scale the Data to Make the NN easier to converge
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(train_data)
# Transform the training and testing data
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

verify_data=scaler.transform(verify_data)

#----------------------------------------------------------------
#Apply the MLPClassifier:
#----------------------------------------------------------------
acc_array = [0] * 5
seed=6
for s in range (1,seed):
    #Init MLPClassifier
    # clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(64,64),
    #                     activation='tanh', learning_rate_init=0.02,max_iter=2000,random_state=s)

    #Using stochastic gradient descent
    # clf = MLPClassifier(solver='sgd', alpha=0.01,hidden_layer_sizes=(64),
    #                     activation='logistic', learning_rate_init=0.01,max_iter=2000,random_state=s)

    clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(64,64),
                        activation='relu', learning_rate_init=0.01,max_iter=5000,random_state=s)

    #Fit the Model
    result = clf.fit(train_data,train_data_label)
    #Predict
    prediction = clf.predict(test_data)
    #Get Accuracy
    acc = accuracy_score(test_data_label, prediction)
    #Store in the Array
    acc_array[s-1] = acc
#----------------------------------------------------------------
#Fetch & Print the Results:
#----------------------------------------------------------------
#     print(classification_report(test_data_label,prediction))
#     print("Accuracy using MLPClassifier and Random Seed:",s,":",str(acc))
#     print(confusion_matrix(test_data_label,prediction))
print("Mean Accuracy using MLPClassifier Classifier: ",np.array(acc_array).mean())

verify_prediction=clf.predict(verify_data)
print("Verify test data personal:---------------------------")
print(classification_report(verify_data_label,verify_prediction))
print(confusion_matrix(verify_data_label,verify_prediction))
print("end---------------------------")
#----------------------------------------------------------------
# Init the Models for Comparision
#----------------------------------------------------------------

#
# models = [BaggingClassifier(), RandomForestClassifier(), AdaBoostClassifier(),
#           KNeighborsClassifier(),GaussianNB(),tree.DecisionTreeClassifier(),
#           svm.SVC(kernel='linear', C=1), OutputCodeClassifier(BaggingClassifier()),
#             OneVsRestClassifier(svm.SVC(kernel='linear'))]
#
# model_names = ["Bagging with DT", "Random Forest", "AdaBoost", "KNN","Naive Bayes","Decision Tree",
#                "Linear SVM","OutputCodeClassifier with Linear SVM" ,"OneVsRestClassifier with Linear SVM"]
# #----------------------------------------------------------------
# # Run Each Model
# #----------------------------------------------------------------
# for model,name in zip(models,model_names):
#     model.fit(data_train, label_train)
#     # Display the relative importance of each attribute
#     if name == "Random Forest":
#         print(model.feature_importances_)
#     #Predict
#     prediction = model.predict(data_test)
#     # Print Accuracy
#     acc = accuracy_score(label_test, prediction)
#     print("Accuracy Using",name,": " + str(acc)+'\n')
#     print(classification_report(label_test,prediction))
# print(confusion_matrix(label_test, prediction))
