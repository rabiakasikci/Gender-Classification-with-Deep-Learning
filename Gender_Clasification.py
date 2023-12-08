# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:42:00 2023

@author: Rabia KAŞIKCI
"""
#import library

import glob
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import plot_model



#This data obtain from Kaggle: 
#https://www.kaggle.com/datasets/gmlmrinalini/manwomandetection/data


#Functions
def image_read(data, label):
    for i in data:
        try:
            data_images = np.array(Image.open(i).resize((width, height)))
            all_data.append(data_images)
            data_label.append(label)
        except Exception as e:
            print(f"Error: Could not read file {i}. Error message: {e}")

def my_train_test_split(all_data, data_label, test_size, random_state):
    return train_test_split(all_data, data_label, test_size=test_size, random_state=random_state)


def plot_and_save_gender_distribution(man_data, women_data, save_path):
    number_of_man, number_of_women = len(man_data), len(women_data)
    sns.set(style="darkgrid")
 
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(x=['Men', 'Women'], y=[number_of_man, number_of_women], palette="bright", ax=ax)

    plt.ylabel("Number of gender")
    plt.title("Number of woman/man in data")
    



    plt.savefig(save_path)
    plt.show()

def plot_and_save_accuracy_and_loss(epochs, train_loss, train_accuracy,save_path):
    # Plot training loss and accuracy
    plt.figure(figsize=(12, 10))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
def data_visualization(num_of_woman_in_train,num_of_man_in_train,num_of_train,num_of_woman_in_test,num_of_man_in_test,num_of_test):
    class_labels = ['Woman in Train', 'Man in Train','Train', 
                    'Woman in Test', 'Man in Test','Test']
    class_counts = [num_of_woman_in_train, num_of_man_in_train,num_of_train, num_of_woman_in_test, num_of_man_in_test,num_of_test]

    # Bar plot çizimi
    plt.figure(figsize=(12, 10))
    plt.bar(class_labels, class_counts, color=['red', 'green', 'blue', 'purple','grey','orange'])
    plt.xlabel('Class Labels')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(rotation=90)
    plt.savefig("Count_of_Data.png")
    plt.show()
   
def data_count(data,label):
    num_of_woman = np.count_nonzero(data == label)
    num_of_man = np.count_nonzero(data == label)
    return num_of_man,num_of_woman


#Data Read
   
man_path = glob.glob('C:\\Users\\USER\\Desktop\\Deep_Learning\\Gender_Clasification\\Data_Set\\man\\*')
women_path = glob.glob('C:\\Users\\USER\\Desktop\\Deep_Learning\\Gender_Clasification\\Data_Set\\woman\\*')

data_label = []
all_data = []


width, height = 64, 64

image_read(man_path, 0)
image_read(women_path, 1)

all_data, data_label = np.array(all_data), np.array(data_label)

#reshape data size
max_size = max(max(image.shape) for image in all_data)

all_data_resized = np.array([np.pad(image, ((0, max_size - image.shape[0]), (0, max_size - image.shape[1]), (0, 0)), mode='constant', constant_values=0) for image in all_data])
all_data, data_label = np.array(all_data_resized), np.array(data_label)
all_data = all_data / 255.0
x_train, x_test, y_train, y_test = my_train_test_split(all_data, data_label, 0.33, 0)

#Deep Learning
#Create Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(max_size, max_size, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32)


# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


#Data visualization

plot_and_save_gender_distribution(x_train, x_test, 'Gender_distribution.png')


# Visualize training metrics
history = model.fit(x_train, y_train, epochs=10, batch_size=32)
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
epochs = range(1, len(train_loss) + 1)
plot_and_save_accuracy_and_loss(epochs, train_loss, train_accuracy,'Accuracy_and_loss_graph.png')


#Confusion Matrix

threshold=0.5
y_pred = model.predict(x_test)
y_pred_classes = (y_pred > threshold).astype(int)  
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Man', 'Predicted Woman'], yticklabels=['Actual Man', 'Actual Woman'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig("Confution_Matrix")
plt.show()




#Data_Count
num_of_woman_in_test,num_of_woman_in_train=data_count(y_train,1)
num_of_man_in_test,num_of_man_in_train=data_count(y_train,0)
num_of_train=len(y_train)
num_of_test=len(y_test)
data_visualization(num_of_woman_in_train,num_of_man_in_train,num_of_train,num_of_woman_in_test,num_of_man_in_test,num_of_test)


# Modelin yapısını görselleştir
plot_model(model, to_file='Deep_Learnin_model_plot.png', show_shapes=True, show_layer_names=True)


#Extra
#Show Pictures
def images_show(image_name):
    # Görüntüyü gösterin
    imgplot = plt.imshow(image_name)
    plt.show()

for i in range(0,2):
    print(man_path[i])
    images_show(Image.open(man_path[i]))
 
    
