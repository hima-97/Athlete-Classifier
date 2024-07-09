# Athlete Classifier

## Project Description

This project involves implementing an AI program that predicts the performance of Olympic athletes based on various features. The program uses a Naive Bayes Classifier to classify athletes' performance with an accuracy of 77.5%.

- **High performance** athletes are labeled as `1`.
- **Low performance** athletes are labeled as `0`.

### Data Files

- `training.txt`: Used to train the Naive Bayes Classifier.
- `testing.txt`: Used to test the accuracy of the classifier.

### Data Format

Each data file consists of an arbitrary number of rows and 12 columns, separated by commas:
1. `age`
2. `gender`
3. `height_cm`
4. `weight_kg`
5. `body_fat_%`
6. `diastolic`
7. `systolic`
8. `grip_force`
9. `sit_and_bend_forward_cm`
10. `sit_up_count`
11. `broad_jump_cm`
12. `label` (only in `training.txt`)

After processing the `testing.txt` file, the program prints a label (`0` or `1`) for each athlete's performance and the overall accuracy.

## Implementation Details

The program is implemented in Python and depends on the `pandas` library. The Naive Bayes Classifier is used to classify athletes' performance based on selected features. The classifier assumes that the features are conditionally independent given the class.

### Key Steps

1. **Load Data**: Read the training and testing data using `pandas`.
2. **Preprocess Data**: Select the features `sit_and_bend_forward_cm` and `sit_up_count` based on their higher correlation with the performance label.
3. **Model Training**: Calculate the mean and variance for each feature for each class.
4. **Model Evaluation**: Calculate log-probability scores for test features and determine predictions based on the highest score.
5. **Accuracy Calculation**: Compute and print the accuracy of the predictions.

### Improvements and Suggestions

To further improve the accuracy:
- Segregate the dataset by gender and train separately.
- Exclude irrelevant attributes.
- Weigh attributes differently based on their influence.

## Tools and Concepts

- **Languages**: Python
- **Libraries**: `pandas`
- **Concepts**: Artificial Intelligence (AI), Naive Bayes Classifier, Data Preprocessing, Log-Likelihood Calculation

## Running and Testing the Project

1. Clone the repository to your local machine.
2. Ensure you have the `pandas` library installed:
   ```bash
   pip install pandas


Run the project with the following command:
python NaiveBayesClassifier.py training.txt testing.txt