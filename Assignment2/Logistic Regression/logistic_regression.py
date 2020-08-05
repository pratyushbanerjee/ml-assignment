#L Regression
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import statistics

#importing the dataset
image_detection=pd.read_csv('data_banknote_authentication_excel.csv')

#splitting the data into testing and training data
test=image_detection.sample(274)
train = image_detection[~image_detection.isin(test)]
train.dropna(inplace = True)

#defining sigmoid function,the loss function 
def sigmoid(x):
    return 1/(1+np.exp(-x))
def square_loss(x,y):
    z=y-x
    return np.mean(pow(z,2))
def scale(x):
    mean=statistics.mean(x)
    variance=statistics.variance(x)
    for i in range(0,len(x)):
        x[i]=(x[i]-mean)/(variance)
    return x
scaled=scale(image_detection.entropy)

#creating testing and training variables, as well as the dependant(class) and the independant variables(entropy)

x_tr,y_tr=train.entropy,train['class']
x_te,y_te=test.entropy,test['class']

#Implementing Gradient Descent algorithm
lr = 0.01 #learning late
const=np.random.uniform(0,1)
W =const+np.random.uniform(0,1) # colom 1
b = 0.1
for i in range(10000):
    z = np.dot(x_tr, W) + b
y_pred = sigmoid(z)
l = square_loss(y_pred, y_tr)
gradient_W = np.dot((y_pred-y_tr).T, x_tr)/x_tr.shape[0]
gradient_b = np.mean(y_pred-y_tr)
W = W-lr * gradient_W
b = b-lr* gradient_b

#implementing the sigmoid function 
for i in range(len(x_te)):
    r = sigmoid(np.dot(x_te, W)+b)

#filling up the model results in the class_1 list
class_1=[]
for i in range(0,len(r)):
    if r[i]<0.5:
        class_1.append(0)
    else:
        class_1.append(1)

#number of zeroes and ones according to our model        
nummodel_1=0
nummodel_2=0
for i in range(0,len(class_1)):
    if class_1[i]==0:
        nummodel_1=nummodel_1+1
    else:
        nummodel_2=nummodel_2+1
#number of atual zeroes and ones in the dataset
a=test['class']
numoriginal_1=0
numoriginal_2=0
for i in a:
    if i==0:
        numoriginal_1=numoriginal_1+1
    else:
        numoriginal_2=numoriginal_2+1

        
#Finding out their F Score and Accuracy
TP=0
TN=0
FP=0
FN=0

if (nummodel_2>numoriginal_2):
    FP=nummodel_2-numoriginal_2
    FN=0
    TP=numoriginal_2
    TN=nummodel_1
else:
    FN=nummodel_1-numoriginal_1
    FP=0
    TN=numoriginal_1
    TP=nummodel_2
    
accuracy= (TP+TN)/(TN+TP+FP+FN)
precision=TP/(TP+FP)
recall=TP/(TP+FN)
F_score=2*precision*recall/(precision+recall)

#L_1 Regression
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import statistics

#importing the dataset
image_detection=pd.read_csv('C:/Users/hp/Desktop/data_banknote_authentication_excel.csv')

#splitting the data into testing and training data
test=image_detection.sample(274)
train = image_detection[~image_detection.isin(test)]
train.dropna(inplace = True)

#defining sigmoid function,the loss function 
def sigmoid(x):
    return 1/(1+np.exp(-x))
def square_loss(x,y):
    z=y-x
    return np.mean(pow(z,2))
def scale(x):
    mean=statistics.mean(x)
    variance=statistics.variance(x)
    for i in range(0,len(x)):
        x[i]=(x[i]-mean)/(variance)
    return x
scaled=scale(image_detection.entropy)

#creating testing and training variables, as well as the dependant(class) and the independant variables(entropy)

x_tr,y_tr=train.entropy,train['class']
x_te,y_te=test.entropy,test['class']

#Implementing Gradient Descent algorithm
lr = 0.01 #learning late
const=np.random.uniform(0,1)
W =const+np.random.uniform(0,1) # colom 1
L1_coeff=5
b = 0.1
for i in range(10000):
    z = np.dot(x_tr, W) + b
y_pred = sigmoid(z)
l = square_loss(y_pred, y_tr)
gradient_W = np.dot((y_pred-y_tr).T, x_tr)/x_tr.shape[0]+L1_coeff*np.sign(W)
gradient_b = np.mean(y_pred-y_tr)
W = W-lr * gradient_W
b = b-lr* gradient_b

#implementing the sigmoid function 
for i in range(len(x_te)):
    r = sigmoid(np.dot(x_te, W)+b)

#filling up the model results in the class_1 list
class_1=[]
for i in range(0,len(r)):
    if r[i]<0.5:
        class_1.append(0)
    else:
        class_1.append(1)

#number of zeroes and ones according to our model        
nummodel_1=0
nummodel_2=0
for i in range(0,len(class_1)):
    if class_1[i]==0:
        nummodel_1=nummodel_1+1
    else:
        nummodel_2=nummodel_2+1
#number of atual zeroes and ones in the dataset
a=test['class']
numoriginal_1=0
numoriginal_2=0
for i in a:
    if i==0:
        numoriginal_1=numoriginal_1+1
    else:
        numoriginal_2=numoriginal_2+1

        
#Finding out their F Score and Accuracy
TP=0
TN=0
FP=0
FN=0

if (nummodel_2>numoriginal_2):
    FP=nummodel_2-numoriginal_2
    FN=0
    TP=numoriginal_2
    TN=nummodel_1
else:
    FN=nummodel_1-numoriginal_1
    FP=0
    TN=numoriginal_1
    TP=nummodel_2
    
accuracy= (TP+TN)/(TN+TP+FP+FN)
precision=TP/(TP+FP)
recall=TP/(TP+FN)
F_score=2*precision*recall/(precision+recall)

#L_2 Regression
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import statistics

#importing the dataset
image_detection=pd.read_csv('C:/Users/hp/Desktop/data_banknote_authentication_excel.csv')

#splitting the data into testing and training data
test=image_detection.sample(274)
train = image_detection[~image_detection.isin(test)]
train.dropna(inplace = True)

#defining sigmoid function,the loss function 
def sigmoid(x):
    return 1/(1+np.exp(-x))
def square_loss(x,y):
    z=y-x
    return np.mean(pow(z,2))
def scale(x):
    mean=statistics.mean(x)
    variance=statistics.variance(x)
    for i in range(0,len(x)):
        x[i]=(x[i]-mean)/(variance)
    return x
scaled=scale(image_detection.entropy)

#creating testing and training variables, as well as the dependant(class) and the independant variables(entropy)

x_tr,y_tr=train.entropy,train['class']
x_te,y_te=test.entropy,test['class']

#Implementing Gradient Descent algorithm
lr = 0.01 #learning late
const=np.random.uniform(0,1)
W =const+np.random.uniform(0,1) # colom 1
L2_coeff=5
b = 0.1
for i in range(10000):
    z = np.dot(x_tr, W) + b
y_pred = sigmoid(z)
l = square_loss(y_pred, y_tr)
gradient_W = np.dot((y_pred-y_tr).T, x_tr)/x_tr.shape[0]+L2_coeff*2*W
gradient_b = np.mean(y_pred-y_tr)
W = W-lr * gradient_W
b = b-lr* gradient_b

#implementing the sigmoid function 
for i in range(len(x_te)):
    r = sigmoid(np.dot(x_te, W)+b)

#filling up the model results in the class_1 list
class_1=[]
for i in range(0,len(r)):
    if r[i]<0.5:
        class_1.append(0)
    else:
        class_1.append(1)

#number of zeroes and ones according to our model        
nummodel_1=0
nummodel_2=0
for i in range(0,len(class_1)):
    if class_1[i]==0:
        nummodel_1=nummodel_1+1
    else:
        nummodel_2=nummodel_2+1
#number of atual zeroes and ones in the dataset
a=test['class']
numoriginal_1=0
numoriginal_2=0
for i in a:
    if i==0:
        numoriginal_1=numoriginal_1+1
    else:
        numoriginal_2=numoriginal_2+1

        
#Finding out their F Score and Accuracy
TP=0
TN=0
FP=0
FN=0

if (nummodel_2>numoriginal_2):
    FP=nummodel_2-numoriginal_2
    FN=0
    TP=numoriginal_2
    TN=nummodel_1
else:
    FN=nummodel_1-numoriginal_1
    FP=0
    TN=numoriginal_1
    TP=nummodel_2
    
accuracy= (TP+TN)/(TN+TP+FP+FN)
precision=TP/(TP+FP)
recall=TP/(TP+FN)
F_score=2*precision*recall/(precision+recall)
