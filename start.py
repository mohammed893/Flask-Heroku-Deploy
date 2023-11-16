from flask import Flask, render_template, request ,jsonify
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import os
import numpy as np


app = Flask(__name__)

#Define image size
IMG_SIZE = 224
BATCH_SIZE = 32
def process_img (image_path , img_size = IMG_SIZE) :
  """
  TAKE PATH TURN AND INTO TENSOR

  """

  #read in an image file
  image = tf.io.read_file(image_path)
  #Turn jpg into numerical tensor with 3 colour channels Red Green Blue
  image = tf.image.decode_jpeg(image , channels = 3)
  #Convert the color channel value from 0 - 255 to 0 - 1 values  (NORMALIZATION)
  image = tf.image.convert_image_dtype(image , tf.float32)
  #resize image to our desired value (224 , 224)
  image = tf.image.resize(image , size = [img_size , img_size])
  return image
def create_data_batches(x , y = None , batch_size = BATCH_SIZE , valid_data = False , 
                        test_data = False):
  """
  Create Batches of data out of image (X) and label (y) pairs
  Shuffles the data -- TO MAKE SURE ORDER does not affect out model
  DON'T shuffle if it's a validation data
  """
  #if the data is a test data set , we don't have labels
  if test_data:
    print("Creating test data batches")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x))) #only filepath no labels
    data_batch = data.map(process_img).batch(batch_size)
    return data_batch
#If the data is a valid dataset , we don't need to shuffle it
#   elif valid_data:
#     print("Creating Valid data batches")
#     data = tf.data.Dataset.from_tensor_slices((tf.constant(x) , #File_paths
#                                                tf.constant(y))) #Labels
#     data_batch = data.map(get_img_label).batch(batch_size)
#     return data_batch
#   #Train data , we have labels , we have to shuffle
  else:
    # print("Creating Train data batches")
    # #Turn filepaths and labels into Tensors
    # data = tf.data.Dataset.from_tensor_slices((tf.constant(x) , tf.constant(y)))
    # #Shuffling 
    # data = data.shuffle(buffer_size = len(x))
    # #Creating (image , label) tuples (this also turns th e img path into a preprocessed img)
    # data = data.map(get_img_label)

    #Turn the Training data into batches
    data_batch = data.batch(BATCH_SIZE)
    return data_batch
def load_model(model_path):
  """
load a saved model from a path
  """
  print("Loading Saved Model")
  model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer" : hub.KerasLayer})
  return model
model = load_model("20231116-05051700111111-1000.h5")
test_path = "static/"
def ready (path):
    test_filenames = [path]
    test_data = create_data_batches(test_filenames, test_data=True)
    test_predictions = model.predict(test_data, verbose = 0)
    return f"label:{np.argmax(test_predictions[0])}"

# test_filenames = [test_path +fname for fname in os.listdir(test_path)]
# test_data = create_data_batches(test_filenames, test_data=True)
# test_predictions = model.predict(test_data, verbose = 0)
# print(f"label:{np.argmax(test_predictions[0])}")



# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/" + img.filename	
		img.save(img_path) 
		p = ready(img_path)
	return p


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
