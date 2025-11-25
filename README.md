# Titanic Survival Prediction

This project predicts passenger survival on the Titanic using **Logistic Regression**. It is based on the Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic). The code includes data preprocessing, feature scaling, encoding, model training, and submission generation.

## Dataset

* **Train dataset:** `train.csv`
* **Test dataset:** `test.csv`

You can download the dataset from [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic).

## Features Used

* **Numerical features:** Age, Fare
* **Categorical features:** Sex, Embarked
* Dropped features: PassengerId, Name, Ticket, Cabin

## Preprocessing Steps

1. Filled missing values:

   * `Age` → median
   * `Fare` → median (test set only)
   * `Embarked` → mode
2. Scaled numerical features (`Age` and `Fare`) using `StandardScaler`.
3. One-hot encoded categorical features (`Sex` and `Embarked`).

## Model

* **Algorithm:** Logistic Regression
* **Library:** scikit-learn (`LogisticRegression`)

## How to Run

1. Place the `train.csv` and `test.csv` files in the same folder or update the paths in the script.
2. Run the Python script:

```bash
python titanic_logistic_regression.py
```

3. A `submission.csv` file will be generated for Kaggle submission.

## Folder Structure

```
titanic-project/
│
├── train.csv
├── test.csv
├── titanic_logistic_regression.py
└── submission.csv
```

## Notes

* The model currently uses only logistic regression. You can experiment with other classifiers like KNN, Random Forest, or SVM.
* Feature engineering and hyperparameter tuning can further improve the accuracy.

If you want, I can also **write an even shorter “resume-friendly” version** that’s perfect for showing in your GitHub portfolio. Do you want me to do that?
