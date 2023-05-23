import tensorflow as tf
img_width = 200
img_height = 50
def encode_single_img(img_path, label):
    # Read image
    img = tf.io.read_file(img_path)
    # Convert to grayscale to reduce size (not cause information loss)
    img = tf.io.decode_png(img, channels=1)
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
        res = tf.strings.reduce_join(int_to_char(res)).numpy().decode('utf-8')
        output_text.append(res)
    return output_text


ocr_model = tf.keras.models.load_model('ocr_model')
ocr_model.summary()
pred = decode_batch_prediction()