import os
import sys
import numpy as np
import pandas as pd

# Naive Bayes rule:
# P(class|data) = (P(data|class) * P(class)) / P(data)
# Where P(class|data) is the probability of class given the provided data.

# Rather than attempting to calculate the probabilities of each attribute value, they are assumed to be conditionally independent given the class value.

# Opening the training.txt file from 1st command line argument:
training_file = open(sys.argv[1], 'r')

# Using for loop to read each line of the training.txt file:
count = 0
for line in training_file:
    count += 1
    print("{}".format(line.strip()))
 
# Closing training.txt file:
training_file.close()

# We will need to calculate the probability of data by the class they belong to, the so-called base rate.
# This means that we will first need to separate our training data by class.