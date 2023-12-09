import numpy as np
import streamlit as st

from keras.preprocessing.image import img_to_array
from tensorflow import keras
from traingenerator import train_generator
from PIL import Image
from googletrans import Translator

def load_model(filename):
  model = keras.models.load_model(filename)
  return model

def load_image():
  upload_image = st.file_uploader('Выберите изображение', type=['jpg', 'jpeg', 'png'], label_visibility='hidden')
  if upload_image is not None:
    image = Image.open(upload_image)
    st.image(image, caption='Ваше изображение', use_column_width=True)

    if st.button("Распознать!"):
      input_image = preprocess_image(image, model)

      translator = Translator()
      translation = translator.translate(input_image, dest='ru')

      st.write(f'Вид бабочки с Вашего изображения: {translation.text}')

def preprocess_image(image_path, model):
  img = image_path.resize((128, 128))
  img_arr = img_to_array(img)
  img_arr = img_arr/255
  arr = np.array([img_arr])
  arr.shape
  prediction = predict_with_model(model, arr)

  return prediction

def predict_with_model(model, image_array):
  preds = model.predict(image_array)
  print(preds)
  class_labels = list(train_generator.class_indices.keys())
  decoded_labels = [class_labels[np.argmax(pred)] for pred in preds]

  return decoded_labels[0]


model = load_model('model.h5')

st.title('Что это за бабочка?')
st.subheader('Вы можете загрузить изображение бабочки, и при помощи нашей нейронной сети узнать её вид')
st.text('Загрузите изображение бабочки в формате: jpg, jpeg, png')

img = load_image()





