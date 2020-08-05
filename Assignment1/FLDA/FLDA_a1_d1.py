#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing the dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d = pd.read_csv('a1_d1.csv', header=None)


#Finding means of different classes

X_d = []
Y_d = []
pos = []
neg = []


x, y = d.shape
for i in range(x):
    d1 = d.iloc[i][:2]
    t1 = d.iloc[i][2]
    X_d.append(d1)
    Y_d.append(t1)
    if t1==1:
        pos.append(d1)
    else:
        neg.append(d1)
    
X_d = np.array(X_d, dtype='float32')
Y_d = np.array(Y_d, dtype='int32')
pos = np.array(pos, dtype='float32')
neg = np.array(neg, dtype='float32')

mean1_n = np.mean(neg, axis=0)
mean1_p = np.mean(pos, axis=0)

#Difference in means of the classes
diff = abs(mean1_p - mean1_n)

#Finding Sw-1 by calculating deviation from mean

x, y = pos.shape
s1 = np.zeros((y, y))

for i in range(x):
    a1 = pos[i] - mean1_p
    b1 = a1.reshape(y, 1)
    a1 = a1.reshape(1, y)
    s1 = s1 + np.dot(b1, a1)

x, y = neg.shape
s2 = np.zeros((y, y))

for i in range(x):
    a1 = neg[i] - mean1_n
    b1 = a1.reshape(y, 1)
    a1 = a1.reshape(1, y)
    s2 = s2 + np.dot(b1, a1)

s = s1 + s2
s = np.linalg.inv(s)
diff.resize(y, 1)
final = np.dot(s, diff)

#Plotting the points in 2-d
plt.style.use('ggplot')

plt.scatter(pos[:, 0], pos[:, 1])
plt.scatter(neg[:, 0], neg[:, 1])

y = (final[1]/final[0])*x + 140
plt.plot(x, y, '-g')

plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()


final = np.dot(s, diff)
slope = -(final[0]/final[1])
new_pos = []
x, y = pos.shape
for i in range(x):
    d = pos[i][1] - slope*pos[i][0]   
    X = -slope*((d-140)/(slope**2 + 1))
    Y = (-140*(slope**2) + 2*d*(slope**2) + d)/(slope**2 + 1)
    new_pos.append([X[0], Y[0]])
    
new_pos = np.array(new_pos)

#Finding value of w and transforming dataset in 1 dimension

slope = -(final[0]/final[1])
new_neg = []
x, y = neg.shape
for i in range(x):
    d = neg[i][1] - slope*neg[i][0]
    X = -slope*((d-140)/(slope**2 + 1))
    Y = (-140*(slope**2) + 2*d*(slope**2) + d)/(slope**2 + 1)
    new_neg.append([X[0], Y[0]])
    
new_neg= np.array(new_neg)

#Finding Intersection point

p = (140*slope)/(slope**2 + 1)
q = p*slope

#Plotting the dataset in one dimension with intersection point

plt.scatter(pos[:, 0], pos[:, 1])
plt.scatter(neg[:, 0], neg[:, 1])
plt.scatter(new_pos[:, 0], new_pos[:, 1])
plt.scatter(new_neg[:, 0], new_neg[:, 1])

plt.scatter(p, q, color='blue')

plt.plot(x, (final[1]/final[0])*x + 140, '-g')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()

#Finding the accuracy and Fscore

acpos = 0
acneg = 0
correct = 0
totpos = 0
totneg = 0

x, y = new_pos.shape
for i in range(x):
    if new_pos[i][1] > q[0]:
        acpos = acpos + 1
        correct = correct + 1
        
x, y = new_neg.shape
for i in range(x):
    if new_neg[i][1] < q[0]:
        acneg = acneg + 1
        correct = correct + 1
      
x, y = X_d.shape
for i in range(x):
    if X_d[i][1] > q[0]:
        totpos = totpos + 1
    else:
        totneg = totneg + 1
        
accuracy = correct / x * 100
precision = acpos / totpos
x, y = new_pos.shape
recall = acpos / x
fscore = (2*precision*recall)/(precision + recall)
fscore


# In[ ]:




