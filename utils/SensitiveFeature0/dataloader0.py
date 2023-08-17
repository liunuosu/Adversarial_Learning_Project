import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer


def dataloader0():
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                     'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week',
                    'native-country', 'income']

    data_train = pd.read_csv("data/adult_data.csv", header=None)
    data_train.columns = column_names
    data_test = pd.read_csv("data/adult_test.csv", header=None)
    data_test.columns = column_names

    data_train = data_train.dropna()
    data_test = data_test.dropna()

    # Bucketize age, assign a bin number to each age
    data_train['age'] = np.digitize(data_train['age'], bins=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    data_test['age'] = np.digitize(data_test['age'], bins=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Drop fnlwgt column
    data_train = data_train.drop(columns=['fnlwgt'])
    data_test = data_test.drop(columns=['fnlwgt'])

    # Replace income with booleans, Space before 50k is crucial
    data_train['income'] = (data_train['income'] == ' >50K').astype(int)
    data_test['income'] = (data_test['income'] == ' >50K.').astype(int)

    target_train = data_train[["income", "gender"]].copy()
    target_train.replace([' Male', ' Female'], [1, 0], inplace=True)

    target_test = data_test[["income", "gender"]].copy()
    target_test.replace([' Male', ' Female'], [1, 0], inplace=True)

    dataset_train = data_train.drop("income", axis=1)
    dataset_test = data_test.drop("income", axis=1)

    numvars = ['education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'age']
    categorical = dataset_train.columns.difference(numvars)

    preprocessor = make_column_transformer(
        (StandardScaler(), numvars),
        (OneHotEncoder(handle_unknown='ignore'), categorical)
    )
    dataset_train = preprocessor.fit_transform(dataset_train)
    dataset_test = preprocessor.transform(dataset_test)

    return dataset_train,  dataset_test, target_train, target_test, numvars, categorical

