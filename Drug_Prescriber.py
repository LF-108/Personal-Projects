# Drug Prediction Algorithm
import warnings
import pandas as pd
import random
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from numpy import array

# Ignores any "unconsequential" errors that may occur in running the model
warnings.filterwarnings('ignore')

data = pd.read_csv('/Users/larafernandes/PycharmProjects/Drug_Prescriber/drug200.csv')

# Holds the attributes that influence which drug is prescribed
data_dummy = pd.get_dummies(data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']], drop_first=True)

# An initial look at the data
print(data.head())

# Holds the drugs
data_dummy['Drug'] = data['Drug']

# Preliminary Analysis of the data

# Count of each drug generally - We can see DrugY is the most prescribed
print(data['Drug'].value_counts())
print("");

# Distribution of Drug by Sex - From this it appears as if Sex does not influence which drug is prescribed
# greatly
sex_drug_count = data.groupby(['Drug', 'Sex'])['Drug'].count()
print(sex_drug_count)
print("");

# Distribution of Drug by BP - From the table it appears BP is a prominent feature, with those with
# High BP only taking drugs Y, A, B, those with Low only taking C, X, and Y, and those with Normal levels only
# taking Y and X
BP_drug_count = data.groupby(['Drug', 'BP'])['Drug'].count()
print(BP_drug_count)
print("");

# Distribution of Drug by Cholesterol seems to have some influence, with only those with High Cholesterol taking
# drugC. Outside of this, the split between normal and high cholesterol for each drug is relatively even
chol_drug_count = data.groupby(['Drug', 'Cholesterol'])['Drug'].count()
print(chol_drug_count)
print("");

# Distribution of different BP levels by Sex
sex_BP_count = data.groupby(['Sex', 'BP'])['BP'].count()
print(sex_BP_count)
print("");

# Distribution of different cholesterol levels by Sex
sex_chol_count = data.groupby(['Sex', 'Cholesterol'])['Cholesterol'].count()
print(sex_chol_count)
print("");

# The average Na_to_K level by Drug. This seems to be a prominent feature, due to the significantly
# higher levels in those who are prescribed drug Y
print("Na_to_K levels by Drug:")
NA_K_drug = data.groupby('Drug')['Na_to_K'].mean()
print(NA_K_drug)
print("");

# The testing set, that holds the instances we will be testing the model on
Y = data_dummy['Drug']

# The training set, which holds the instances the model will learn to predict from
X = data_dummy.drop('Drug', axis=1)

# Will store the accuracy of the 10 runs, and be used to calculate the mean, variance, and standard deviation
mean = []

#Will store the most influential factor determined by each trial
top_factor = []
#Will store the 3 most influential factors determined by each trial
top_three_factors = []
# Since these sets are needed for later analysis outside of the loop below, we will define them here
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=35, shuffle=True)

for x in range(10):
    # Chooses a random number between 20 and 50 for the random_state
    rand_state = random.randint(20, 50)

    # Splits the the data into A training set X_train, a testing set, X_test, and their respective
    # Y values, Y_train and Y_test. For maximum variability, both shuffle and random_state are used
    # to scramble the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=rand_state, shuffle=True)

    # Sets up the Gaussian classifier that will train the model and predict
    gaussian_NB = GaussianNB()

    # Trains the model based on the training data
    gaussian_NB.fit(X_train, Y_train)

    # Predicts results based on training above given X_test as testing data
    testing_set_prediction = gaussian_NB.predict(X_test)

    # Gives the accuracy of the model based on comparing the true results, Y_test, with the results
    # the model predicted above
    model_accuracy = metrics.accuracy_score(Y_test, testing_set_prediction)

    # Appends the accuracy to the list mean for later analysis
    mean.append(model_accuracy)
    print('Model Accuracy for trial ' + str(x + 1) + ' is {0:0.4}'.format(model_accuracy))

    # Which feature influences results the most - since we want the most prominent feature, k = 1 for one feature
    strongest_feature = SelectKBest(score_func=chi2, k=1)

    # Similar to the above fit method used to train the model, we use X_train and Y_train to find the most
    # defining feature
    select = strongest_feature.fit_transform(X_train, Y_train)

    # filters out the strongest attribute from the list of attributes present in the data set
    index = strongest_feature.get_support()

    # creates an array out of the data columns sans the Drug column
    attributes = array(data.columns)

    top_factor.append(attributes[index])

    # Which feature influences results the most - since we want the top 3 features, k is three this time
    strongest_feature = SelectKBest(score_func=chi2, k=3)

    select_3 = strongest_feature.fit_transform(X_train, Y_train)

    index_3 = strongest_feature.get_support()

    attributes_3 = array(data.columns)

    top_three_factors.append(attributes_3[index_3])


print(' ')
# Gets the mean accuracy of 10 trials
print('Mean Accuracy {0:0.4}'.format(np.mean(mean)))

# Gets the variance of the data
print('Variance: {0:0.4}'.format(np.var(mean)))

# Gets the standard deviation of 10 trials
print('Standard Deviation: {0:0.4}'.format(np.std(mean)))

# Prints out the most influential feature by picking a random element from the list we've collected from the trials
# Surprisingly using the SelectKBest method, Sex was shown to be the most influential feature
# Considering that sex can influence concentrations of various hormones, BP, and so many other
# factors, it does make some sense that sex is the most influential feature, as it would influence
# many of the other chemical based features that influence the prescription of the drug

top_index = random.randrange(len(top_factor))
print(f"The most influential attribute when determining which drug to prescribe is {top_factor[top_index]})")

# Prints out the three most influential features by picking a random element from the list we've collected
# from the trials

top_three_index = random.randrange(len(top_three_factors))
print(f"The three most influential attributes when determining which drug to prescribe are {top_three_factors[top_three_index]})")
