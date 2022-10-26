# imported necessary packages
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings  # current version generates a bunch of warnings that we'll ignore
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import seaborn as sns

warnings.filterwarnings("ignore")

# glass is a dataframe that we load the glass.csv data into.
glass = pd.read_csv("C:/Users/Kiran Kumar Kongari/Desktop/Machine Learning/Dataset/Dataset/glass.csv")

x = glass.iloc[:, :-1].values
y = glass.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Classification using Gaussian Naive Bayes
classifier = GaussianNB()
y_pred = classifier.fit(x_train, y_train).predict(x_test)

# Summary of the predictions made by the Gaussian Naive Bayes classifier
print("Classification using Gaussian Naive Bayes\n")
print("Classification Report : \n", classification_report(y_test, y_pred))
print("Confusion Matrix : \n", confusion_matrix(y_test, y_pred))
# Accuracy score
print('accuracy is', accuracy_score(y_pred, y_test))

# ------------------------------------------------------------------------------------------------------------------------------
# Classification using Linear Support Vector Machine's
classifier = LinearSVC(verbose=0)
y_pred = classifier.fit(x_train, y_train).predict(x_test)

# Summary of the predictions made by the LinearSVC classifier
print("\n-----------------------------------------------------------------------------------------\n"
      "Classification using Linear Support Vector Machine's\n")
print("Classification Report : \n ", classification_report(y_test, y_pred))
print("Confusion Matrix : \n", confusion_matrix(y_test, y_pred))
# Accuracy score
print('accuracy is', accuracy_score(y_pred, y_test))

# -----------------------------------------------------------------------------------------------------------------------------
# Visualization of the correlation using heatmap, histplot, scatterplot
correlation_matrix = glass.corr().round(2)  # Round to 2 decimal places
print("\n Correlation Matrix : \n", correlation_matrix)  # display correlation matrix

# Creating plot
sns.heatmap(data=correlation_matrix, annot=True)  # Set annot = True to print the values inside the squares
# show plot
plt.show()

# Creating plot
sns.histplot(data=correlation_matrix)
# show plot
plt.show()

# Creating plot
sns.scatterplot(data=correlation_matrix)
# show plot
plt.show()




