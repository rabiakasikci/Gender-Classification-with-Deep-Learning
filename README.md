# Gender Classification
A simple project that classifies gender based on facial features using machine learning.

## Setup
You can download and use the file.
Steps to install project dependencies and run the project:
After downloading the file, make sure to update the file path in the `config.py` file to point to the location where your data is stored.
After downloading the file, you can run it with Python3 in the same file.
Or you can type 
`python Gender_Classification.py`
into the terminal you opened in the same file.

## Use
After typing `python Gender_Classification.py` into the terminal, the model will be trained and the accuracy rate will be displayed on the screen. Additionally, five different image files will be saved, showcasing the model's predictions.

### Dependencies
- Python3
- glob
- numpy as np
- PIL
- Image
- pandas
- sklearn.model_selection 
- train_test_split
- tensorflow.keras.models 
- Sequential
- tensorflow.keras.layers 
- Conv2D, MaxPooling2D, Flatten, Dense
- matplotlib
- pyplot 
- seaborn
- sklearn.metrics
- confusion_matrix
- tensorflow.keras.utils
- plot_model