import numpy
from sklearn import neighbors, model_selection, naive_bayes
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv(r'/Path/')
X = numpy.array(df.iloc[:,0:df.shape[1]-1])
y = numpy.array(df.iloc[:,df.shape[1]-1:df.shape[1]])
y = y.flatten()


for i in range(1,20):
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    scores = model_selection.cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    scores = numpy.mean(scores)
    print(scores)

def printScores(model, X, y):
    scores = model_selection.cross_val_score(model, X, y, cv=10, scoring='accuracy')
    predict = model_selection.cross_val_predict(model, X, y, cv=10)
    print("Accuracy: " + str('{0:.3f}'.format(numpy.mean(scores))))
    print("Precision: " + str('{0:.3f}'.format(precision_score(y, predict, average="macro"))))
    print("Recall: " + str('{0:.3f}'.format(recall_score(y, predict, average="macro"))))
    print("F1: " + str('{0:.3f}'.format(f1_score(y, predict, average="macro"))))

print("\nknn:")
printScores(neighbors.KNeighborsClassifier(n_neighbors=5), X, y)
print("\nnb:")
printScores(naive_bayes.GaussianNB(), X, y)
print("\nrf:")
printScores(RandomForestClassifier(), X, y)
