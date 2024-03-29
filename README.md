# Project Description

This project consists of implementing an artificial intelligence (AI) program that automatically predicts the performance of different olympic athletes based on a series of features. This program allows the user to predict with an accuracy of 77.5% which athletes will perform well and which ones won't. <br>

A Naive Baise Classifier is constructed to accurately classify each athlete's performance. <br>
Athletes with high performance grade are labeled 1 and athletes with not high perforance grade are labeled 0. <br>

A `training.txt` dataset file is used to train the Naive Bayes Classifier to accurately predict a athlete's performance. <br>
A `testing.txt` dataset file is then used to test the accuracy of the Naive Bayes Classifier. <br>

Each of these files has an arbitrary number of rows and 12 columns. <br>
Columns are separated by commas, with the first 11 columns as features and the 12th as the performance label.  <br>
The name of the 11 columns in order are the following: age, gender, height_cm, weight_kg, body fat_%, diastolic, systolic, grip_force, sit_and_bend_forward_cm, sit_up_count, broad_jump_cm.
The target column is named class. <br>

After testing the `testing.txt` dataset file a label (0 or 1) is printed for each athlete's performance, one label per line. At the end, the program also prints the accuracy (77.5%) of the Naive Bayes Classifier (i.e. prediction).

# Implementation Details:

The program depends on `pandas` and its dependencies. <br>
The code contains no functions and no classes. <br>

The program uses the Naive Bayes rule, which states that: <br>
the probability of a class given the provided data (P(class|data)) is equal to the product of the probability of the data given the class (P(data|class)) and the probability of the class (P(class)), divided by the probability of the data (P(data)). <br>

To accomplish this, the program first reads in a training dataset file called "training.txt" and a testing dataset file called "testing.txt" using the pandas library, and then performs minimal pre-processing on the data. The data is first cleaned by adding column names and then by selecting only the "sit_and_bend_forward_cm" and "sit_up_count" columns, which were found to have a higher correlation with the label than the other columns. This step is important because it prevents overfitting and makes the model more robust. <br>

The program then models the selected columns as two independent univariate Gaussian random variables, which can be described by their mean and variance. These are computed by using APIs from the pandas library, after the data is separated by its class label. <br>

The program then uses the training data to train the classifier and test the accuracy by using testing data set, also it computes the log-likelihood for each column for each class, summing the log-likelihood for all columns for each class and then comparing the log-likelihood for the two classes. The accuracy of the Naive Bayes classifier is also computed and printed. <br>

The program also suggests some ideas to improve the accuracy of the classifier, such as further segregating the data set into male and female, excluding some athlete's attributes, and giving more importance to those that influence more an athlete's class. <br>

Overall, this project demonstrates a solid understanding of machine learning concepts and implementation using python, as well as an understanding of how to work with data using pandas library. It also shows an ability to pre-process data, train and test a machine learning model, and make predictions based on the data. <br> <br> <br>

Besides reading from the files through the pandas API, the data is minimally pre-processed by:
- Add a column name to data frame
- Select "sit_and_bend_forward_cm" and "sit_up_count" columns only, by discarding the rest

The two columns are cherry-picked from an exploratory data analysis. <br>
By looking at the correlation matrix, notice how the correlation between these two columns with the 
label is significantly higher than all the rest. <br>
The other columns have negligible correlation, so they are dropped from the data to avoid overfitting. <br>
To summarize, each instance is represented by its feature columns only. <br>

Both "sit_and_bend_forward_cm" and "sit_up_count" are modeled as two independent univariate Gaussian random variables.
Note that the independent part is a constraint of the Naive Bayes model. <br>
A Gaussian random variable can be uniquely described by its mean and variance, both which can be computed by using two APIs from the `pandas` library. <br>
Note how the mean and the variances for thee variables are computed after the data is separated by its class label. <br>

To compute the result, we need to first test the data, which is done in the following steps:
- Compute the log-likelihood for each column for each class
- Sum the log-likelihood for all column for each class
- Compare the log-likelihood for the two classes
- Compare the accuracy

Only the relative magnitude between the likelihood values are considered here.<br>
Summing the Log-likelihood value consists of simply multiplying them, because of the Naive Bayes conditional independence assumption. <br>

Some ideas to improve the accuracy of the Naive Bayes Classifier could be:
- Further segregate the data set into male and female, and then train independently
- Some athletes' attributes might be useless and therefore could be excluded to predict their class
- Weigh the athletes' attributes differently, giving more importance to those that influence more an athlete's class

# Tools and Concepts
You can run and test the full project by running the following command:
- Languages: Python
- VSCode
- Artificial intelligence (AI)
- Naive Bayes Classifier

# Running and Testing the Project

You can run and test the full project by running the following command:
- `python NaiveBayesClassifier.py training.txt testing.txt`

You can also test your own `testing.txt` dataset file formatted in the same way as the given testing file.