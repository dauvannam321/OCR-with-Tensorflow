import streamlit as st
import tensorflow as tf
import pdf2image
from PIL import Image, ImageOps
import numpy as np
img_width = 200
img_height = 50
max_len = 5
int_to_char = ['[UNK]',
 '2',
 '3',
 '4',
 '5',
 '6',
 '7',
 '8',
 'b',
 'c',
 'd',
 'e',
 'f',
 'g',
 'm',
 'n',
 'p',
 'w',
 'x',
 'y']
def encode_single_img(img):
    # Read image
    # img = tf.io.read_file(img_path)
    # Convert to grayscale to reduce size (not cause information loss)
    # img = tf.io.decode_png(img, channels=1)
    # img = tf.image.rgb_to_grayscale(img)
    # Convert to float32 and scale to [0, 1]
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize
    img = tf.image.resize(img, [img_height, img_width])
    # Transpose 
    img = tf.transpose(img, perm=[1, 0, 2])
    return img
def decode_batch_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_len]
    output_text = []
    for res in results:
        
        array = res.numpy()
        chars = [int_to_char[i] for i in array]
        res = tf.strings.reduce_join(chars).numpy().decode('utf-8')
        
        output_text.append(res)
    return output_text

ocr_model = tf.keras.models.load_model('ocr_predict')

st.title('Trang web đọc Captcha')
uploaded = st.file_uploader('Chọn ảnh Captcha muốn đọc')
if uploaded is not None:
    img = Image.open(uploaded)
    img = ImageOps.grayscale(img)
    st.image(img)
    img = tf.keras.utils.img_to_array(img).astype('uint8')  
    img = tf.convert_to_tensor(img)
   
    img = encode_single_img(img)
    
    img = np.expand_dims(img, axis=0)

    pred = ocr_model.predict(img)

    output = decode_batch_prediction(pred)
    st.write(output)
    st.write('Success')

