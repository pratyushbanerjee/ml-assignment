#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

d = pd.read_csv('a1_d2.csv', header=None)

#Finding means of different classes

X_d = []
Y_d = []
pos = []
neg = []


x, y = d.shape
for i in range(x):
    d1 = d.iloc[i][:3]
    t1 = d.iloc[i][3]
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

mean1_p = np.mean(pos, axis=0)
mean1_n = np.mean(neg, axis=0)

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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#Plotting data in D Dimension

ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='y', marker='o')
ax.scatter(neg[:, 0], neg[:, 1], neg[:, 2], c='b', marker='v')

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

a,b,c,d = final[0][0], final[1][0], final[2][0], 0

x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)

X,Y = np.meshgrid(x,y)
Z = (d - a*X - b*Y) / c


fig = plt.figure()

surf = ax.plot_surface(X, Y, Z)

xp = 1
yp = 2

a,b,c,d = final[0][0], final[1][0], final[2][0], 0
zp = (d - a*xp - b*yp) / c

t = np.linspace(-80, 80, 40)
xnew = a*t + xp
ynew = b*t + yp
znew = c*t + zp

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

a,b,c,d = final[0][0], final[1][0], final[2][0], 0

x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)

X,Y = np.meshgrid(x,y)
Z = (d - a*X - b*Y) / c

# Plot the surface

surf = ax.plot_surface(X, Y, Z)
ax.plot3D(xnew, ynew, znew, 'red')

plt.show()

x, y = pos.shape
finalpos = []

for i in range(x):
    d = a*pos[i][0] + b*pos[i][1] + c*pos[i][2]
    t = -(a*xp + b*yp + c*zp + d)/(a**2 + b**2 + c**2)
    xx = a*t + xp
    yy = b*t + yp
    zz = c*t + zp
    finalpos.append([xx, yy, zz])
finalpos = np.array(finalpos)

x, y = neg.shape
finalneg = []

for i in range(x):
    d = a*neg[i][0] + b*neg[i][1] + c*neg[i][2]
    t = -(a*xp + b*yp + c*zp + d)/(a**2 + b**2 + c**2)
    xx = a*t + xp
    yy = b*t + yp
    zz = c*t + zp
    finalneg.append([xx, yy, zz])
finalneg = np.array(finalneg)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(finalpos[:, 0], finalpos[:, 1], finalpos[:, 2], c='y', marker='o')
ax.scatter(finalneg[:, 0], finalneg[:, 1], finalneg[:, 2], c='b', marker='v')
ax.scatter(xp, yp, zp, c='r', marker='v')

plt.show()

#Finding Fscore

x, y = finalpos.shape
correct = 0
acpos = 0
for i in range(x):
    if finalpos[i][2] < zp:
        acpos = acpos + 1
        correct = correct + 1
        
x, y = finalneg.shape
acneg = 0
for i in range(x):
    if finalneg[i][2] > zp:
        correct = correct + 1
        acneg = acneg + 1
        
totpos, _ = pos.shape
totneg, _ = neg.shape

accuracy = correct / (totpos + totneg) * 100
precision = acpos / totpos
recall = acpos / totpos
fscore = (2*recall*precision) / (recall + precision)

fscore


# In[ ]:




