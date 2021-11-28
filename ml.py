import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


class ML():
    def __init__(self, path):
        self.df = pd.read_csv(path, delim_whitespace=True)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self._X = None
        self._y = None

        self._prepare_data()

    def _prepare_data(self):
        self._X = self.df.iloc[:, :-1]
        self._y = self.df.iloc[:, -1:]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self._X, self._y, test_size=0.33, random_state=42)

    def knn(self):
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(self.X_train, self.y_train.values.ravel())
        y_pred = knn.predict(self.X_test)

        return accuracy_score(self.y_test, y_pred)

    def random_forest(self):
        forest = RandomForestClassifier(random_state=0)
        forest.fit(self.X_train, self.y_train.values.ravel())
        y_pred = forest.predict(self.X_test)

        importances = forest.feature_importances_

        return accuracy_score(self.y_test, y_pred), importances

    def decision_tree(self):
        trees = DecisionTreeClassifier(random_state=0)
        trees.fit(self.X_train, self.y_train.values.ravel())
        y_pred = trees.predict(self.X_test)

        importances = trees.feature_importances_

        return accuracy_score(self.y_test, y_pred), importances

    def logistic_regression(self):
        regression = LogisticRegression(solver='lbfgs', max_iter=10000)
        regression.fit(self.X_train, self.y_train.values.ravel())
        y_pred = regression.predict(self.X_test)

        return accuracy_score(self.y_test, y_pred)

    def gaussian_nb(self):
        gaussian = GaussianNB()
        gaussian.fit(self.X_train, self.y_train.values.ravel())
        y_pred = gaussian.predict(self.X_test)

        return accuracy_score(self.y_test, y_pred)

    def display(self):
        labels = [
            "KNN",
            "Random Forest",
            "Decision Tree",
            "Logistic Regression",
            "GaussianNB"
        ]
        knn_acc = self.knn()
        rf_acc, rf_importances = self.random_forest()
        dt_acc, dt_importances = self.decision_tree()
        lr_acc = self.logistic_regression()
        gnb_acc = self.gaussian_nb()

        values = [
            knn_acc,
            rf_acc,
            dt_acc,
            lr_acc,
            gnb_acc
        ]

        fig, axs = plt.subplots(3, 1, figsize=(12, 8))
        axs[0].bar(labels, values, color='maroon',
                   width=0.4)
        axs[0].set_ylabel("Accuracy")
        
        axs[0].set_title("Classifiers comparision")

        axs[1].bar(self._X.columns, rf_importances, color='limegreen',
                   width=0.4)
        axs[1].set_title("Random forest - feature importance")
        axs[1].set_ylabel("%")

        axs[2].bar(self._X.columns, dt_importances, color='limegreen',
                   width=0.4)
        axs[2].set_title("Decision Tree - feature importance")
        axs[2].set_ylabel("%")

        fig.tight_layout()
        plt.show()


ml = ML('australin2.csv')
ml.display()
