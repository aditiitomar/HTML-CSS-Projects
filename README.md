
# ML-driven-Diabetes-Detection

This project aims to predict diabetes using machine learning techniques. The prediction is based on a dataset containing various health parameters of individuals, such as glucose level, blood pressure, and age.

### Requirements

- Python 3.x
- Libraries: pandas, numpy, scikit-learn
- Diabetes dataset [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
  
### Data Preprocessing

The first step involves data preprocessing:

- Standardize the data to a common range.
- Split the dataset into training and testing sets.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
diabetes_dataset = pd.read_csv('/path/to/diabetes.csv')

# Separating features and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize the data
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(standardized_data, Y, test_size=0.2, random_state=2)
```

### Training

Train the Support Vector Machine (SVM) model using the training data:

```python
from sklearn import svm
from sklearn.metrics import accuracy_score

# Initialize SVM Classifier
classifier = svm.SVC(kernel='linear')

# Train the model
classifier.fit(X_train, Y_train)

# Evaluate training data accuracy
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy Score of training data: ', training_data_accuracy)
```

### Testing

Test the trained model using the testing data:

```python
# Evaluate test data accuracy
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy Score of test data: ', test_data_accuracy)
```

### Prediction

Finally, using the trained model to make predictions on new data:

```python
# Example input data
input_data = (10, 168, 74, 0, 0, 38, 0.537, 34)

# Convert input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the input data
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data
std_data = scaler.transform(input_data_reshaped)

# Make prediction
prediction = classifier.predict(std_data)

# Display prediction
if prediction[0] == 0:
    print('Not Diabetic')
else:
    print('Diabetic')
```
