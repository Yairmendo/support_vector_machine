import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from sklearn import svm
from sklearn.datasets import make_blobs

#  Aquí se divide en hiperplanos nuestros datos
#  en este caso son 2 hiperplanos (centers es igual a 2)
X, y = make_blobs(n_samples=100,centers=2,random_state=6)

#  clf es clasiffier
clf = svm.SVC(kernel='linear',C=1000)
clf.fit(X, y)
#  Scatter establece la dispersión de los datos
plt.scatter(X[:, 0], X[:, 1], c=y,s=30,cmap=plt.cm.Paired)

#  Aquí se establecieron los ejes "x" , "y"
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

#  Aquí se establecieron los límites y escalas
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
#  Se define la malla de datos para que esten ordenados
YY, XX = np.meshgrid(yy, xx)
#  Se distribuyen los datos en la malla
xy = np.vstack([XX.ravel(), YY.ravel()]).T
#  Aquí está la función de decisión
Z = clf.decision_function(xy).reshape(XX.shape)

#  Esto lo hace más estético
ax.contour(XX, YY, Z, colors='g',levels=[-1,0, 1], alpha=0.5,linestyles=['-.', '-', '-.'])
ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=100,linewidth=1,facecolors='none')
#  Nos muestra la gráfica
plt.show()