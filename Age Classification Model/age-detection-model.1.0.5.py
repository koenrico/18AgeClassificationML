# Description: Age classification model implementation
# Author: Frederik Koen

# Notes: Remember to change the dataset directory to use UTKFace or FG-Net
#        Remember to chnage the keras model for transfer learning to use DenseNet201 or VGG19

import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf

from keras.layers import Rescaling
from tensorflow.keras.applications import DenseNet201
#from tensorflow.keras.applications import VGG19
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Setup strategy to run on on multiple GPU's
strategy = tf.distribute.MirroredStrategy()

# Set Autotune variable
AUTOTUNE = tf.data.AUTOTUNE

# Define common variables
batch_size=64
img_width=224
img_height=224
num_classes = 1
seed = 123

# Define image data directory
# NOTE: Change directory to load UTKFace or FG-Net
dataset_dir = r'/home/ctext/Desktop/rico/AgeEstimation/utkface_18_preprocessed/'

# Load the training data (70% of the data), set a random seed to reproduce results and suffle the data
training_dataset = image_dataset_from_directory(
  dataset_dir + 'training',
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  shuffle=True)

# Load the training data (20% of the data), set a random seed to reproduce results and suffle the data
validation_dataset = image_dataset_from_directory(
  dataset_dir + 'validation',
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  shuffle=True)

# Load the training data (10% of the data), set a random seed to reproduce results
testing_dataset = image_dataset_from_directory(
  dataset_dir + 'testing',
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=64)

# Get the class names from the training dataset, it will be the same for the validation and testing data sets
class_names = training_dataset.class_names

# Prefetch the data to increase performance
training_dataset = training_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# Setup data augmentation, flipping, ratation, zooming, translations, and contrast
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.1),
  tf.keras.layers.RandomZoom(0.1),
  tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
  tf.keras.layers.RandomContrast(factor=0.1),
])

# Setup preprocessing required by the keras models
# Note: Change to the appropriate preprocessing for the densenet and vgg19 models
preprocess_input = tf.keras.applications.densenet.preprocess_input

# Setup resaceling layer
scale_layer = Rescaling(scale=1./225)

# Set the base learning rate
base_learning_rate = 0.0001

# Build and compile the model within the stratagy to allow the use of multiple GPU's when training
with strategy.scope():
    # Build the base model and remove the top layer of the transfer learning model
    base_model = DenseNet201(input_shape=(img_width,img_height,3),include_top=False,weights='imagenet')
    # Freeze all the layers in the base model to skip training in these layers
    base_model.trainable = False
    # Setup the model inputs
    inputs = tf.keras.Input(shape=(img_width,img_height,3))
    # Apply data augmentation
    x = data_augmentation(inputs)
    # Apply preprocessing
    x = preprocess_input(x)
    # Add the base model
    x = base_model(x,training=False)
    # Add a Max Pooling layer
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    # Flatten the output from the previous layer
    x = tf.keras.layers.Flatten(name='flatten')(x)
    # Add a dropout layer
    x = tf.keras.layers.Dropout(0.2)(x)
    # Add a fully connected layer with 1024 connections
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    # Add the final fully connected layer and set the number of connections to the number of classes
    outputs = tf.keras.layers.Dense(num_classes)(x)
    # Build the model
    model = tf.keras.Model(inputs,outputs)
    # Print a summary of the model
    model.summary()
    # Complile the final model
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), metrics=['accuracy'])

# Add a callback function to stop training if validatin loss is stabilising
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=3)

# Set the initial epochs to 20
initial_epochs = 20

# Start training the model
history = model.fit(training_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset,
                    callbacks=[callback])

# Get the training results from the model and plot on a line graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('results-acc-initial.png') 

# Fine-tune the model within the stratagy to allow the use of multiple GPU's when training
with strategy.scope():
    # Unfreeze base model layers to allow training of the layers
    base_model.trainable = True
    # Complile the fine tuned model, and set the learning rate lower
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
               optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
               metrics=['accuracy'])

# Set the fine tune epochs to 20
fine_tune_epochs = 20
# Set the total number of epochs, initial + fine tuned
total_epochs =  initial_epochs + fine_tune_epochs

# Start training on the fine tuned model
history_fine  = model.fit(training_dataset, 
                       validation_data=validation_dataset,
                       initial_epoch=history.epoch[-1],
                       epochs=total_epochs,
                       callbacks=[callback])  

# Get the training results from the model and plot on a line graph
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('results-acc-fine.png')  

# Get the image and label batches from the testing dataset
dataset_batch = testing_dataset.map(lambda x, y: (x, y))
image_batch, labels_batch = next(iter(dataset_batch))

# Final evaluation of the model
scores = model.evaluate(testing_dataset, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Get the model predictions
preds = model.predict(image_batch,verbose=1)

# Calculate the MAE in regression based on the predictions of the model
mae = tf.keras.losses.MeanAbsoluteError()
print("MAE: ", mae(labels_batch, preds).numpy())

# Draw a ROC curve of the model predictions
fpr, tpr, _ = roc_curve(labels_batch, preds)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig("roc")
plt.show()