import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time
from PIL import Image

#---------------------------------------------------------------
unique_breeds = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
        'american_staffordshire_terrier', 'appenzeller',
        'australian_terrier', 'basenji', 'basset', 'beagle',
        'bedlington_terrier', 'bernese_mountain_dog',
        'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
        'bluetick', 'border_collie', 'border_terrier', 'borzoi',
        'boston_bull', 'bouvier_des_flandres', 'boxer',
        'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
        'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
        'chow', 'clumber', 'cocker_spaniel', 'collie',
        'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
        'doberman', 'english_foxhound', 'english_setter',
        'english_springer', 'entlebucher', 'eskimo_dog',
        'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
        'german_short-haired_pointer', 'giant_schnauzer',
        'golden_retriever', 'gordon_setter', 'great_dane',
        'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
        'ibizan_hound', 'irish_setter', 'irish_terrier',
        'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
        'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
        'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
        'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
        'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
        'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
        'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
        'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
        'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
        'saint_bernard', 'saluki', 'samoyed', 'schipperke',
        'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
        'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
        'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
        'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
        'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
        'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
        'west_highland_white_terrier', 'whippet',
        'wire-haired_fox_terrier', 'yorkshire_terrier']

#---------------------------------------------------------------
def process_image(x, img_size=224):
  '''
  Takes an image file path and turn the image into a Tensor
  '''
  # Read in an image file
  # image = tf.io.read_file(path)
  # Turn the jpg image into numerical Tensor with 3 color channels
  image = tf.image.decode_jpeg(x, channels=3)
  # Convert the color channel values from 0-255 to 0-1 values
  image = tf.image.convert_image_dtype(image, tf.float32) # this is called Normalization
  # Resize the image to our desired value (224,224)
  # image = tf.image.resize(image, size=[img_size, img_size])
  image = tf.image.resize(image,size=[224,224])
  image = tf.expand_dims(image, axis=0)
  return image

#---------------------------------------------------------------
# Turn prediction probabilities into their respective label (easier to understand)
def get_pred_label(prediction_probabilities):
  '''
  Turns an array of prediction probabilites into a label.
  '''
  return unique_breeds[np.argmax(prediction_probabilities)]
#----------------------------------------------------------------



if __name__ == '__main__':
    # ---------------------------------------------------------------
    with st.sidebar:
        dog = Image.open('pet.png')
        st.image(dog)
        st.header(':red[Project Dog Vision]')
        st.text("""
        It's a deep learning project
        inspired from 
        a Kaggle Competition 
        `Dog Breed Identification`.

        'Transfer Learning' 
        has been implemented using 
        two different models
        `mobilenet_v2` & `resnet_v2_50'
        from TensorFlow Hub. 
        These model are trained on
        image dataset from competition. 
        
        """)
        st.caption(':red[:copyright: Abhijeet Kamble]')
        st.link_button('LinkedIn', url='https://www.linkedin.com/in/abhijeetk597/')
        st.link_button('GitHub', url='')

    st.header('Upload photo of your dog :dog:')
    file = st.file_uploader('')
    # ---------------------------------------------------------------
    if file is not None:
        data = process_image(file.getvalue())
        output_image = file.getvalue()
        # --------------------------------------------------------------
        # Load model
        model_1 = tf.keras.models.load_model('models/dog-vision-mobilenetv2.h5',
                                             custom_objects={'KerasLayer': hub.KerasLayer})
        model_2 = tf.keras.models.load_model('models/dog-vision-resnet_v2-Adam.h5',
                                             custom_objects={'KerasLayer': hub.KerasLayer})
        # Make predictions on the custom data
        pred_1 = model_1.predict(data)
        pred_2 = model_2.predict(data)
        accuracy_1 = round(np.max(pred_1) * 100, 2)
        accuracy_2 = round(np.max(pred_2) * 100, 2)
        # Get custom image prediction labels
        pred_label_1 = get_pred_label(pred_1)
        pred_label_2 = get_pred_label(pred_2)
        # -------------------------------------------------
        bar = st.progress(50)
        time.sleep(3)
        bar.progress(100)
        # ------------------------------------------------------------
        st.subheader("Output :point_down:")
        t1, t2 = st.tabs(['mobilenet_v2', 'resnet_v2_50'])
        # t1.title('Model: mobilenet_v2')     # t2.title('Model: resnet_V2_50')
        t1.caption('This model is trained directly on available images from competition dataset')
        t2.caption('This model is trained on randomly flipping images from competition dataset')
        t1.title(pred_label_1)
        t2.title(pred_label_2)
        t1.text('Prediction confidence %: ')
        t2.text('Prediction confidence %: ')
        t1.write(accuracy_1)
        t2.write(accuracy_2)
        # ---------------------------------------------------------------
        st.image(output_image)
        # ---------------------------------------------------------------
        st.caption(':blue[:sparkling_heart: Crafted with curiosity, passion and purpose to bring ideas into life :rocket:]')
