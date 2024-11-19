# Logistic Regression

## Overview
The program performs logistic regression on a binary classification dataset. The key steps involved are:
1. **Cross Validation**: The dataset is divided into a training set and a validation set. The model is trained on the training set and tested on the validation set.
2. **Handling Missing Values**: Address any `NaN` values in the training set, converting them into numerical values.
3. **Label Adjustment**: Labels in the dataset that are $-1$ are changed to $0$ for binary classification.
4. **Data Augmentation**: Duplicate the minority positive examples in the training set several times, add them to the training set, and then shuffle the data.
5. **Model Fitting**: Employ a logistic funtion to fit the model and use the negative log likelihood as the cost function for stochastic gradient descent.
6. **Regularization**: Regularization techniques are applied to the cost function to prevent overfitting.
7. **Model Evaluation**: The model's accuracy and F1 score are computed on the validation dataset.
8. **Model and Logs**: The trained model and its parameters are saved in the `model` folder, while logs are recorded in the `log` folder for each run.

## Usage
1. Place the dataset CSV file in the `dataset` folder.
2. Run `python run.py`

### Adam
```
python run.py --adam=True
```

### SGD with momentum
```
python run.py --sgd_with_moment=True
```
