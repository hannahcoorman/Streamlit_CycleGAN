import streamlit as st

import numpy as np
import pandas as pd
import matplotlib as plt
import tensorflow as tf
import cv2

from tensorflow import keras

import io
from PIL import Image
from scipy.ndimage.filters import median_filter

AUTOTUNE = tf.data.AUTOTUNE

# Constants
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

#=======================================
##############FUNCTIONS#################
#=======================================

# Resize the images to equal size
@st.cache()
def resize_image(image):
	RESIZE_WIDTH = 256
	RESIZE_HEIGHT = 256
	imgs = []
	x1 = 0
	x2 = 0
	y1 = 0
	y2 = 0
	width = int(image.shape[1])
	height = int(image.shape[0])    
	# Setting the points for cropped image
	if width > height:
	      x1 = int((width - height)/2)      
	      y1 = height
	      x2 = int((width - height)/2 + height)
	      y2 = 0
	      crop_im = image[y2:y1, x1:x2]
	      resized = cv2.resize(crop_im, (RESIZE_HEIGHT,RESIZE_WIDTH), interpolation = cv2.INTER_AREA)
	      resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
	if width < height:
	      x1 = 0
	      y1 = int((height - width)/2 + width)
	      x2 = width
	      y2 = int((height - width)/2)
	      crop_im = image[y2:y1, x1:x2]
	      resized = cv2.resize(crop_im, (RESIZE_HEIGHT,RESIZE_WIDTH), interpolation = cv2.INTER_AREA)
	      resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
	if width == height: 
	      resized = cv2.resize(im, (RESIZE_HEIGHT,RESIZE_WIDTH), interpolation = cv2.INTER_AREA)
	      resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
	return resized



# Generate images by the generator
@st.cache()
def generate_images(model, test_input):
  prediction = model(test_input)
  prediction=prediction.numpy()
  prediction = prediction[0]* 0.5 + 0.5
  return prediction

### IN CASE THE IMAGE IS UPLOADED, PREPROCESSING IS NEEDED TO PUT THE IMAGE IN THE CORRECT FORMAT (not used here)
# normalizing the images to [-1, 1]
@st.cache()
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

# Preprocessing the test images
@st.cache()
def preprocess_image_test(image, label):
  image = normalize(image)
  return image

#=======================================
###########STREAMLIT OUTPUT#############
#=======================================

# Title of Streamlit application
st.title('CycleGAN for Candy Crush Themes')


# Add a selectbox to the sidebar IMAGE:
selected_image = st.sidebar.selectbox(
    'Choose an image',
    ('Summer', 'Winter', 'Accessories', 'No accessories')
)

# Show chosen image
st.write("Chosen image")

# Load chosen preprocessed image
if selected_image == 'Summer':
	st.image('./summer.png')
	test_summer = np.load("C:/Users/hanna/OneDrive/Documenten/2020-2021/Thesis/code/Streamlit/S.npy")
if selected_image == 'Winter':
	st.image('./winter.png')
	test_summer = np.load("C:/Users/hanna/OneDrive/Documenten/2020-2021/Thesis/code/Streamlit/W.npy")
if selected_image == 'Accessories':
	st.image('./accessories.png')
	test_summer = np.load("C:/Users/hanna/OneDrive/Documenten/2020-2021/Thesis/code/Streamlit/A.npy")
if selected_image == 'No accessories':
	st.image('./noaccessories.png')
	test_summer = np.load("C:/Users/hanna/OneDrive/Documenten/2020-2021/Thesis/code/Streamlit/NA.npy")

# Add a selectbox to the sidebar MODEL:
selected_model = st.sidebar.selectbox(
    'Choose a model',
    ('Summer to winter', 'Winter to summer', 'Add accessories', 'Remove accessories')
)

# Add a selectbox to the sidebar LAMBDA:
selected_lambda = st.sidebar.selectbox(
    'Choose a value for lambda',
    ('2', '10', '50')
)

# Load pretrained model
if selected_model == 'Summer to winter':
	if selected_lambda == '2':
		model = tf.keras.models.load_model('./S2W_2')
	if selected_lambda == '10':
		model = tf.keras.models.load_model('./S2W')
	if selected_lambda == '50':
		model = tf.keras.models.load_model('./S2W_50')
if selected_model == 'Winter to summer':
	if selected_lambda == '2':
		model = tf.keras.models.load_model('./W2S_2')
	if selected_lambda == '10':
		model = tf.keras.models.load_model('./W2S')
	if selected_lambda == '50':
		model = tf.keras.models.load_model('./W2S_50')
if selected_model == 'Add accessories':
	if selected_lambda == '2':
		model = tf.keras.models.load_model('./acc_2')
	if selected_lambda == '10':
		model = tf.keras.models.load_model('./acc')
	if selected_lambda == '50':
		model = tf.keras.models.load_model('./acc_50')
if selected_model == 'Remove accessories':
	if selected_lambda == '2':
		model = tf.keras.models.load_model('./acc_2')
	if selected_lambda == '10':
		model = tf.keras.models.load_model('./noacc')
	if selected_lambda == '50':
		model = tf.keras.models.load_model('./acc_50')

	
# Display in & output
st.write("Output of the cycleGAN")

if selected_image == 'Summer':
	inp = cv2.imread('./summer.png',1)
if selected_image == 'Winter':
	inp = cv2.imread('.winter.png',1)
if selected_image == 'Accessories':
	inp = cv2.imread('./accessories.png',1)
if selected_image == 'No accessories':
	inp = cv2.imread('./noaccessories.png',1)

# Processing steps before putting the image in the generator
inp = resize_image(inp)

pred=generate_images(model, test_summer[0])


left_column, right_column = st.beta_columns(2)

left_column.image(inp, caption = "Model Input 256x256")
right_column.image(pred,  caption = "Predicted Image")