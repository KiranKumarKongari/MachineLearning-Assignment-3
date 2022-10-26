# 1. (Titanic Dataset)
#  1. Find the correlation between ‘survived’ (target column) and ‘sex’ column for the Titanic use case in class.
#         a. Do you think we should keep this feature?
#  2. Do at least two visualizations to describe or show correlations.
#  3. Implement Naive Bayes method using scikit-learn library and report the accuracy


import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt

titanic = pd.read_csv("C:/Users/Kiran Kumar Kongari/Desktop/Machine Learning/Dataset/Dataset/train.csv")
print(titanic)

# Finding the correlation between ‘survived’ (target column) and ‘sex’ column

# To get a correlation we need to convert our categorical features to numerical ones.
# Of course the choice of order will affect the correlation but luckily all of our categories seem to be binary
titanic['Survived'] = titanic['Survived'].astype('category').cat.codes
titanic['Sex'] = titanic['Sex'].astype('category').cat.codes

# Used corr() function to find the correlation
correlation_Value = titanic['Sex'].corr(titanic['Survived'])
print("\nThe correlation between ‘survived’ (target column) and ‘sex’ column is : ", correlation_Value)

# ------------------------------------------------------------------------------------------------------------------------------
# a. Do you think we should keep this feature?
#    Yes we should keep this because a large negative correlation is just as useful as a large positive correlation.
#   The only difference is that for a positive correlation, as the feature increases, the target will increase.
#   For a negative correlation, as the feature decreases, the target will increase.

# ------------------------------------------------------------------------------------------------------------------------------
# Visualization of the correlation using heatmap, histplot, scatterplot

# Dropping the unnecessary columns
data = titanic.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns')

# Removing all the data that has missing values
processed_data = data.dropna(axis=0)

# Converts categorical values in the 'Sex' column into numerical values.
data1 = pd.get_dummies(processed_data, drop_first=True)

# Converting the datatype to Float
data1["Survived"] = data1["Survived"].astype(float)
data1["Pclass"] = data1["Pclass"].astype(float)
data1["Sex"] = data1["Sex"].astype(float)

# Calculated the correlation matrix using corr() function
correlation_matrix = data1.corr().round(2)  # Round to 2 decimal places
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

# From the correlation matrix , we can observe that 'Pclass' and 'Fare' have a correlation of -0.55.
# This suggests that these feature pairs are strongly correlated to each other.
# Considering multicollinearity, let's drop the 'Fare' column since it has lower correlation
# with 'Survived' compared to 'Pclass'.
final_df = data1.drop('Fare', axis=1)

# ------------------------------------------------------------------------------------------------------------------------------
# Classification using Gaussian Naive Bayes
x = final_df.drop('Survived', axis=1)
y = final_df['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
classifier = GaussianNB()
y_pred = classifier.fit(x_train, y_train).predict(x_test)

# Summary of the predictions made by the Gaussian Naive Bayes classifier
print("\nClassification using Gaussian Naive Bayes")
print("Classification Report : \n", classification_report(y_test, y_pred))
print("Confusion Matrix : \n", confusion_matrix(y_test, y_pred))

# Accuracy score
print('accuracy is', accuracy_score(y_pred, y_test))
