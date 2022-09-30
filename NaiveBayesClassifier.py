import sys
import math
import pandas as pd

# Naive Bayes rule:
# 		P(class|data) = (P(data|class) * P(class)) / P(data)
# Where P(class|data) is the probability of a class given the provided data.

# Rather than attempting to calculate the probabilities of each attribute value,
# they are assumed to be conditionally independent given the class value.
# Therefore, we need to calculate the probability of data by the class they belong to, the so-called base rate.
# This means that we first need to separate our training data by class.

# These are given column names:
column_names = ["age", "gender", "height_cm", "weight_kg", "body fat_%","diastolic", "systolic", "grip_force", "sit_and_bend_forward_cm", "sit_up_count", "broad_jump_cm", "label"]

# These features are hand picked:
features = ["sit_and_bend_forward_cm","sit_up_count"]

# Step 1 - Reading training file:
# Opening the training.txt file from 1st command line argument:
training_file = open(sys.argv[1], 'r')
training_file = open(sys.argv[1], 'r')
train = pd.read_csv(training_file, header=None)
train.columns = column_names

# Step 2 - Seggregate the data by label:
train_0_M = train[(train["label"] == 0) & (train["gender"] == 'M')]
train_1_M = train[(train["label"] == 1) & (train["gender"] == 'M')]
train_0_F = train[(train["label"] == 0) & (train["gender"] == 'F')]
train_1_F = train[(train["label"] == 1) & (train["gender"] == 'F')]

# Step 3 - Training by computing the mean and variance for each feature:
m_model_mean_0 = train_0_M[features].mean()
m_model_variance_0 = train_0_M[features].var()
m_model_mean_1 = train_1_M[features].mean()
m_model_variance_1 = train_1_M[features].var()
f_model_mean_0 = train_0_F[features].mean()
f_model_variance_0 = train_0_F[features].var()
f_model_mean_1 = train_1_F[features].mean()
f_model_variance_1 = train_1_F[features].var()

# This should give us some idea on class separation:
# print(m_model_mean_0)
# print(m_model_mean_1)
# print(f_model_mean_0)
# print(f_model_mean_1)

# Step 4: Reading testing file
testing_file = open(sys.argv[2], 'r')
test = pd.read_csv(testing_file, header=None)

# Note: unlike the given testing file, the user's testing file won't have the label column.
# This is to make sure not to fail column name assignment.
test.columns = column_names[0:len(test.columns)]

# Step 5 - Evaluate the model score:
test_features = test[features].copy()
for column in features:
	m_mean_0 = m_model_mean_0[column]
	m_variance_0 = m_model_variance_0[column]
	m_mean_1 = m_model_mean_1[column]
	m_variance_1 = m_model_variance_1[column]
	f_mean_0 = f_model_mean_0[column]
	f_variance_0 = f_model_variance_0[column]
	f_mean_1 = f_model_mean_1[column]
	f_variance_1 = f_model_variance_1[column]	
	test_features[column+"_m0"] = test_features[column].apply(lambda x: - math.log(m_variance_0) - ((x - m_mean_0) ** 2)/2/m_variance_0)	
	test_features[column+"_m1"] = test_features[column].apply(lambda x: - math.log(m_variance_1) - ((x - m_mean_1) ** 2)/2/m_variance_1)
	test_features[column+"_f0"] = test_features[column].apply(lambda x: - math.log(f_variance_0) - ((x - f_mean_0) ** 2)/2/f_variance_0)	
	test_features[column+"_f1"] = test_features[column].apply(lambda x: - math.log(f_variance_1) - ((x - f_mean_1) ** 2)/2/f_variance_1)

test_features["score_m0"] = test_features[[x+"_m0" for x in features]].sum(axis=1)
test_features["score_m1"] = test_features[[x+"_m1" for x in features]].sum(axis=1)
test_features["score_f0"] = test_features[[x+"_f0" for x in features]].sum(axis=1)
test_features["score_f1"] = test_features[[x+"_f1" for x in features]].sum(axis=1)

test["m_predicted"] = test_features["score_m1"] > test_features["score_m0"] 
test["m_predicted"] = test["m_predicted"].apply(lambda x: 1 if x else 0)
test["f_predicted"] = test_features["score_f1"] > test_features["score_f0"] 
test["f_predicted"] = test["f_predicted"].apply(lambda x: 1 if x else 0)
test["m"] = test["gender"].apply(lambda x: 1 if x=='M' else 0)
test["f"] = test["gender"].apply(lambda x: 0 if x=='M' else 1)
test["predicted"] = test["m"] * test["m_predicted"] + test["f"] * test["f_predicted"]

# Step 6: Print the result
for predicted in test["predicted"].to_list():
	print(predicted)

# Step 7: Compute accuracy percentage for local testing:
test["accuracy"] = test["label"] == test["predicted"]
test["accuracy"] = test["accuracy"].apply(lambda x: 1 if x else 0)
print(test["accuracy"].mean())