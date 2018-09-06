import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import neighbors, naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    if __name__ == '__main__':
        plt.figure()
        plt.title(title, fontsize = 28)
        plt.tick_params(labelsize =20)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples", fontsize = 23)
        plt.ylabel("Score", fontsize = 23)
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="lower right", prop={'size': 20})
        #mpl.rcParams['figure.figsize'] = [7, 10]
        plt.xticks(np.arange(20, 300, 40))
        #plt.yticks(np.arange(0.3, 1, 0.1))
        plt.tight_layout()
        plt.savefig(title+'.png', dpi=300, orientation='landscape')
        return plt

df = pd.read_csv(r'/Path/')
df['Type'] = df['Type'].astype('str')

X = np.array(df.iloc[:,0:df.shape[1]-1])
y = np.array(df.iloc[:,df.shape[1]-1:df.shape[1]])
y = y.flatten()

# Training data plot.
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

title = "Learning Curves (Naive Bayes)"
estimator = naive_bayes.GaussianNB()
plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4, train_sizes=[range(20, 320, 20)])

title = "Learning Curves (KNN)"
estimator = neighbors.KNeighborsClassifier(n_neighbors=5)
plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4, train_sizes=[range(20, 320, 20)])

title = "Learning Curves (RF)"
estimator = RandomForestClassifier()
plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4, train_sizes=[range(20, 320, 20)])

plt.show()
