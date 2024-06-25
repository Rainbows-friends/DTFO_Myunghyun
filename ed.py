import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Absolute path to the training data directory (use raw string to handle backslashes)
data_dir = r'C:\Users\user\Desktop\AI_test\train'

# Initialize the data generator with rescaling and augmentation parameters
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Define the train generator to load images from the directory
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),  # Resize images to 150x150 pixels
    batch_size=32,  # Number of images to yield per batch
    class_mode='binary'  # Use binary labels for classification
)

# Example of how to use the train generator with model.fit (commented out)
# model.fit(
#     train_generator,
#     steps_per_epoch=2000,
#     epochs=50
# )
