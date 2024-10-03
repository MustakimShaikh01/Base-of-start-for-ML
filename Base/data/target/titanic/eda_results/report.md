# EDA Report

## Summary
- **Shape**: (891, 12)
- **Data Types**:
```
{'PassengerId': dtype('int64'), 'Survived': dtype('int64'), 'Pclass': dtype('int64'), 'Name': dtype('O'), 'Sex': dtype('O'), 'Age': dtype('float64'), 'SibSp': dtype('int64'), 'Parch': dtype('int64'), 'Ticket': dtype('O'), 'Fare': dtype('float64'), 'Cabin': dtype('O'), 'Embarked': dtype('O')}
```
- **Missing Values**:
```
{'PassengerId': 0, 'Survived': 0, 'Pclass': 0, 'Name': 0, 'Sex': 0, 'Age': 177, 'SibSp': 0, 'Parch': 0, 'Ticket': 0, 'Fare': 0, 'Cabin': 687, 'Embarked': 2}
```
- **Basic Statistics**:
```
{'PassengerId': {'count': 891.0, 'unique': nan, 'top': nan, 'freq': nan, 'mean': 446.0, 'std': 257.3538420152301, 'min': 1.0, '25%': 223.5, '50%': 446.0, '75%': 668.5, 'max': 891.0}, 'Survived': {'count': 891.0, 'unique': nan, 'top': nan, 'freq': nan, 'mean': 0.3838383838383838, 'std': 0.4865924542648575, 'min': 0.0, '25%': 0.0, '50%': 0.0, '75%': 1.0, 'max': 1.0}, 'Pclass': {'count': 891.0, 'unique': nan, 'top': nan, 'freq': nan, 'mean': 2.308641975308642, 'std': 0.836071240977049, 'min': 1.0, '25%': 2.0, '50%': 3.0, '75%': 3.0, 'max': 3.0}, 'Name': {'count': 891, 'unique': 891, 'top': 'Dooley, Mr. Patrick', 'freq': 1, 'mean': nan, 'std': nan, 'min': nan, '25%': nan, '50%': nan, '75%': nan, 'max': nan}, 'Sex': {'count': 891, 'unique': 2, 'top': 'male', 'freq': 577, 'mean': nan, 'std': nan, 'min': nan, '25%': nan, '50%': nan, '75%': nan, 'max': nan}, 'Age': {'count': 714.0, 'unique': nan, 'top': nan, 'freq': nan, 'mean': 29.69911764705882, 'std': 14.526497332334042, 'min': 0.42, '25%': 20.125, '50%': 28.0, '75%': 38.0, 'max': 80.0}, 'SibSp': {'count': 891.0, 'unique': nan, 'top': nan, 'freq': nan, 'mean': 0.5230078563411896, 'std': 1.1027434322934317, 'min': 0.0, '25%': 0.0, '50%': 0.0, '75%': 1.0, 'max': 8.0}, 'Parch': {'count': 891.0, 'unique': nan, 'top': nan, 'freq': nan, 'mean': 0.38159371492704824, 'std': 0.8060572211299483, 'min': 0.0, '25%': 0.0, '50%': 0.0, '75%': 0.0, 'max': 6.0}, 'Ticket': {'count': 891, 'unique': 681, 'top': '347082', 'freq': 7, 'mean': nan, 'std': nan, 'min': nan, '25%': nan, '50%': nan, '75%': nan, 'max': nan}, 'Fare': {'count': 891.0, 'unique': nan, 'top': nan, 'freq': nan, 'mean': 32.204207968574636, 'std': 49.6934285971809, 'min': 0.0, '25%': 7.9104, '50%': 14.4542, '75%': 31.0, 'max': 512.3292}, 'Cabin': {'count': 204, 'unique': 147, 'top': 'G6', 'freq': 4, 'mean': nan, 'std': nan, 'min': nan, '25%': nan, '50%': nan, '75%': nan, 'max': nan}, 'Embarked': {'count': 889, 'unique': 3, 'top': 'S', 'freq': 644, 'mean': nan, 'std': nan, 'min': nan, '25%': nan, '50%': nan, '75%': nan, 'max': nan}}
```

## Missing Values
The following columns have missing values:

## Distributions
## Correlation Matrix
![Correlation Matrix](correlation_matrix.png)

## Categorical Features
