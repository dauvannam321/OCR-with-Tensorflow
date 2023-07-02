import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from annotated_text import annotated_text, annotation
import numpy as np
img_width = 200
img_height = 50
max_len = 5
int_to_char = ['[UNK]',
 '1',
 '2',
 '3',
 '4',
 '5',
 '6',
 '7',
 '8',
 '9',
 'A',
 'B',
 'C',
 'D',
 'E',
 'F',
 'G',
 'H',
 'I',
 'J',
 'K',
 'L',
 'M',
 'N',
 'O',
 'P',
 'Q',
 'R',
 'S',
 'T',
 'U',
 'V',
 'W',
 'X',
 'Y',
 'Z',
 'a',
 'b',
 'c',
 'd',
 'e',
 'f',
 'g',
 'h',
 'i',
 'j',
 'k',
 'l',
 'm',
 'n',
 'p',
 'q',
 'r',
 's',
 't',
 'u',
 'v',
 'w',
 'x',
 'y',
 'z']
def int_to_label_decode(array):
    # array = array.numpy()
    res = []
    for i in array:
        if i == -1:
            break
        else:
            res.append(int_to_char[i])
    if res:
        res = tf.strings.reduce_join(res).numpy().decode('utf-8')
        return res        
    else:
        return 'EMPTY'
        
def encode_single_img(img):
    # Convert to grayscale to reduce size (not cause information loss)
    # img = tf.io.decode_png(img, channels=1)
    # img = tf.image.rgb_to_grayscale(img)
    # Convert to float32 and scale to [0, 1]
    # img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    img = img / 255.0
    return img
def decode_batch_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_len]
    output_text = []
    for res in results:
        
        array = res.numpy()
        chars = [int_to_char[i] for i in array]
        res = tf.strings.reduce_join(chars).numpy().decode('utf-8')
        # res = int_to_label_decode(array)

        output_text.append(res)
    return output_text
st.set_page_config(page_title='Trang web đọc Captcha', page_icon='icon.png', layout='wide', initial_sidebar_state='expanded')
ocr_model = tf.keras.models.load_model('model_captcha')
st.sidebar.image('sideimage.png')
col1, col2 = st.columns([5, 1])
col1.title('TRANG WEB ĐỌC CAPTCHA')
col1.subheader('Chọn ảnh muốn đọc (5 ký tự)')
col2.image('question.gif')
uploaded = st.file_uploader('')
if uploaded is not None:
    with st.spinner():
        image = Image.open(uploaded)
        col3, col4, col5 = st.columns(3)
        col4.image(image)
        img = ImageOps.grayscale(image)
        img = tf.keras.utils.img_to_array(img).astype('uint8')
        img = tf.convert_to_tensor(img)
        img = encode_single_img(img)        
        img = np.expand_dims(img, axis=0)
        pred = ocr_model.predict(img)
        output = decode_batch_prediction(pred)
        st.balloons()
        st.divider()
    st.success('Đọc thành công!', icon="✅")
    col6, col7, col8 = st.columns(3)
    with col7:
        annotated_text(
        annotation(output[0], font_family="Comic Sans MS", border="2px dashed", font_size="40px"),
        )

