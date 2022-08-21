# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 15:40:34 2022

@author: kolhe
"""

import streamlit as st
import PIL
import urllib.request
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import cv2
import functools
import os

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

style_urls = dict(
  kanagawa_great_wave='https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg',
  kandinsky_composition_7='https://upload.wikimedia.org/wikipedia/commons/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg',
  hubble_pillars_of_creation='https://upload.wikimedia.org/wikipedia/commons/6/68/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg',
  van_gogh_starry_night='https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg',
  turner_nantes='https://upload.wikimedia.org/wikipedia/commons/b/b7/JMW_Turner_-_Nantes_from_the_Ile_Feydeau.jpg',
  munch_scream='https://upload.wikimedia.org/wikipedia/commons/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg',
  picasso_demoiselles_avignon='https://upload.wikimedia.org/wikipedia/en/4/4c/Les_Demoiselles_d%27Avignon.jpg',
  picasso_violin='https://upload.wikimedia.org/wikipedia/en/3/3c/Pablo_Picasso%2C_1911-12%2C_Violon_%28Violin%29%2C_oil_on_canvas%2C_Kr%C3%B6ller-M%C3%BCller_Museum%2C_Otterlo%2C_Netherlands.jpg',
  picasso_bottle_of_rum='https://upload.wikimedia.org/wikipedia/en/7/7f/Pablo_Picasso%2C_1911%2C_Still_Life_with_a_Bottle_of_Rum%2C_oil_on_canvas%2C_61.3_x_50.5_cm%2C_Metropolitan_Museum_of_Art%2C_New_York.jpg',
  fire='https://upload.wikimedia.org/wikipedia/commons/3/36/Large_bonfire.jpg',
  derkovits_woman_head='https://upload.wikimedia.org/wikipedia/commons/0/0d/Derkovits_Gyula_Woman_head_1922.jpg',
  amadeo_style_life='https://upload.wikimedia.org/wikipedia/commons/8/8e/Untitled_%28Still_life%29_%281913%29_-_Amadeo_Souza-Cardoso_%281887-1918%29_%2817385824283%29.jpg',
  derkovtis_talig='https://upload.wikimedia.org/wikipedia/commons/3/37/Derkovits_Gyula_Talig%C3%A1s_1920.jpg',
  amadeo_cardoso='https://upload.wikimedia.org/wikipedia/commons/7/7d/Amadeo_de_Souza-Cardoso%2C_1915_-_Landscape_with_black_figure.jpg'
)


def crop_center(image):
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None)
def load_image(image_path, name, image_size=(256, 256), preserve_aspect_ratio=True):
  if 'http' in image_path:
    urllib.request.urlretrieve(image_path, "{}.jpg".format(name))
    image = PIL.Image.open("{}.jpg".format(name))
  else:
    image = PIL.Image.open(image_path)
  st.image(image, width=200)
  loc = '{}.jpg'.format(name)
  image.save(loc)

  img = tf.io.decode_image(
      tf.io.read_file(loc),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

st.header("Deep Style Transfer")

# output_image_size
output_image_size = st.number_input("Enter output image size", 256)
output_image_size = int(output_image_size)

# content_img_size
content_img_size = (output_image_size, output_image_size)
style_img_size = (256, 256)  # Recommended to keep it at 256.

# style_image_url

status = st.radio("Select Style Method: ", ('Predefined Styles', 'Enter URL'))
 
# conditional statement to print
# Male if male is selected else print female
# show the result using the success function
if (status == 'Predefined Styles'):
    style_image_key = st.selectbox("Styles: ", list(style_urls.keys()))
    style_image_url = style_urls[style_image_key]
else:
    style_image_url = st.text_input("Enter style image URL", 'https://img.freepik.com/free-vector/violet-fire-colours-hand-painted-background_23-2148427580.jpg?w=2000')


style_image = load_image(style_image_url, 'abstract', style_img_size)
style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

content_image_url = st.text_input("Enter content URL", 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ee/Sai_Pallavi_at_Mca-pre-release-event.jpg/640px-Sai_Pallavi_at_Mca-pre-release-event.jpg')
content_image = load_image(content_image_url, 'content', content_img_size)


#content_image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ee/Sai_Pallavi_at_Mca-pre-release-event.jpg/640px-Sai_Pallavi_at_Mca-pre-release-event.jpg'  # @param {type:"string"}
#style_image_url = 'https://img.freepik.com/free-vector/violet-fire-colours-hand-painted-background_23-2148427580.jpg?w=2000'  
#output_image_size = 384 



if(st.button('Generate Style Transfer')):
    # run model
    result_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    
    cv2.imwrite('generated_img.jpg', cv2.cvtColor(np.squeeze(result_image)*255, cv2.COLOR_BGR2RGB))
    res_img = PIL.Image.open("generated_img.jpg")
    st.image(res_img)
