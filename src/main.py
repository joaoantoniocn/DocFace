import numpy as np
import argparse
from basenet import BaseNetwork
import utils
from lfw import LFWTest
import tensorflow as tf
import time

def euclidian(x1, x2):
  #x2 = x2.transpose()
  x1_norm = np.sum(np.square(x1),  keepdims=True)
  x2_norm = np.sum(np.square(x2),  keepdims=True)
  dist = x1_norm + x2_norm - 2*np.dot(x1,x2)
  return dist

def normalize(x, ord=None, axis=None, epsilon=10e-12):
  ''' Devide the vectors in x by their norms.'''
  if axis is None:
    axis = len(x.shape) - 1
  norm = np.linalg.norm(x, ord=None, axis=axis, keepdims=True)
  x = x / (norm + epsilon)
  return x


def test():
  result = np.load('./result/output.npy')
  # result_l2 = preprocessing.normalize(resrult, norm='l2')
  result_l2 = normalize(result)
  print(euclidian(result_l2[8], result_l2[0]))
  print(euclidian(result_l2[8], result_l2[1]))
  print(euclidian(result_l2[8], result_l2[2]))
  print(euclidian(result_l2[8], result_l2[3]))
  print(euclidian(result_l2[8], result_l2[4]))
  print(euclidian(result_l2[8], result_l2[5]))
  print(euclidian(result_l2[8], result_l2[6]))
  print(euclidian(result_l2[8], result_l2[7]))
  print(euclidian(result_l2[8], result_l2[9]))


# Validar modelo na base de dados LFW.
# É imporante que as imagens a serem validadas tenham sido alinhadas utilizando o MTCNN que pode ser encontrado em '.align/face_detect_align.m'.
# A utilização de qualquer outra forma de alinhamento, variante ou não do MTCNN pode afetar de forma direta na performance do modelo.
# Esse modelo é muito sensivel ao alinhamento das faces, por favor usar o cód em '.align/face_detect_align.m' para alinha-las.
def validate_lfw(args):
  config = utils.import_file(args.config_file, 'config')
  testset = utils.Dataset(args.test_dataset_path)

  # Carregando modelo
  print('Loading network model...')
  network = BaseNetwork()
  network.load_model(args.model_dir)

  # Set up LFW test protocol and load images
  print('Loading images...')
  lfwtest = LFWTest(testset.images)
  lfwtest.init_standard_proto(args.lfw_pairs_file)
  lfwtest.images = utils.preprocess(lfwtest.image_paths, config, is_training=False)

  # Testing on LFW
  print('Testing on standard LFW protocol...')
  embeddings = network.extract_feature(lfwtest.images, config.batch_size)
  accuracy_embeddings, threshold_embeddings = lfwtest.test_standard_proto(embeddings)
  print('Embeddings Accuracy: %2.4f Threshold %2.3f' % (accuracy_embeddings, threshold_embeddings))

def validate_tflite(args):

  # open config files
  config = utils.import_file(args.config_file, 'config')
  testset = utils.Dataset(args.test_dataset_path)

  # Set up LFW test protocol and load images
  print('Loading images...')
  lfwtest = LFWTest(testset.images)
  lfwtest.init_standard_proto(args.lfw_pairs_file)
  lfwtest.images = utils.preprocess(lfwtest.image_paths, config, is_training=False)


  # Load TFLite model and allocate tensors
  print('Loading network tflite model...')
  interpreter = tf.contrib.lite.Interpreter(args.model_dir)
  interpreter.allocate_tensors()


  # Get Input and output tensors
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # ---------------------------------
  # Testing on LFW
  num_images = lfwtest.images.shape[0] if type(lfwtest.images) == np.ndarray else len(lfwtest.images)
  num_features = output_details[0]['shape'][1]

  result = np.ndarray((num_images, num_features), dtype=np.float32)

  print("Extracting Embeddings from LFW...")
  #for start_idx in range(0, num_images, config.batch_size):
  for start_idx in range(num_images):

    # get batch images
    #end_idx = min(num_images, start_idx + config.batch_size)
    #inputs = lfwtest.images[start_idx:end_idx]
    inputs = np.expand_dims(lfwtest.images[start_idx], axis=0)
    #print("input should be: ", str(input_details[0]['shape']))
    #print("input: ", str(inputs.shape))
    # run net
    interpreter.set_tensor(input_details[0]['index'], inputs)
    interpreter.invoke()

    # get batches result
    #result[start_idx:end_idx] = interpreter.get_tensor(output_details[0]['index'])
    result[start_idx] = interpreter.get_tensor(output_details[0]['index'])

    if(start_idx%100 == 0):
      print("Extract " + str(start_idx) + "/" + str(num_images))

  # calculating accuracy and threshold
  print("Calculating accuracy...")
  accuracy_embeddings, threshold_embeddings = lfwtest.test_standard_proto(result)
  print('Embeddings Accuracy: %2.4f Threshold %2.3f' % (accuracy_embeddings, threshold_embeddings))

# -----------------------------
  # Test model on random input data
  #input_shape = input_details[0]['shape']
  #input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
  #interpreter.set_tensor(input_details[0]['index'], input_data)

  #interpreter.invoke()
  #output_data = interpreter.get_tensor(output_details[0]['index'])
  #print(output_data.shape)




# convert a .pb model to a tflite model
# if you have a checkpoint model you will have to convert it on a .pb model first.
# to make this convertion run the code below:
# python3 tensorflow/tensorflow/python/tools/freeze_graph.py --input_meta_graph=DocFace-master/log/faceres_ms/graph.meta --input_checkpoint=DocFace-master/log/faceres_ms/ckpt-320000 --output_graph=DocFace-master/log/model/frozen_graph.pb --output_node_names=outputs --input_binary=True
def convert_model(args):

  # converter o modelo
  input_arrays = ["inputs"]
  output_arrays = ["outputs"]

  converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(args.model_dir, input_arrays, output_arrays)

  # performs the size optimization
  converter.post_training_quantize = True
  tflite_model = converter.convert()

  # writing it out
  open(args.output_file, "wb").write(tflite_model)
  print("Model written at " + str(args.output_file))

def main(args):

  if(args.action == 'validate_lfw'):
    validate_lfw(args)
  elif(args.action == 'convert_model'):
    convert_model(args)
  elif (args.action == 'validate_tflite'):
    validate_tflite(args)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_dir", help="The path to the pre-trained model directory", type=str, default= './log/faceres_ms')
  parser.add_argument("--output_file", help="The path for the output tflite model", type=str,
                      default='./log/tflite/converted_model.tflite')
  parser.add_argument("--test_dataset_path", help="The path to the test dataset, which should already be align using '.align/face_detect_align.m'", type=str,
                      default='./data/lfw_align')
  parser.add_argument("--lfw_pairs_file",
                      help="The path to the LFW standard protocol file",
                      type=str,
                      default='./proto/lfw_pairs.txt')
  parser.add_argument("--config_file",
                      help="The path to the config file",
                      type=str,
                      default='./config/basemodel.py')
  parser.add_argument("--action",
                      help="The action you want to do.",
                      type=str,
                      choices=['validate_lfw', 'convert_model', 'validate_tflite', ''],
                      default='')



  args = parser.parse_args()
  main(args)


# Test the optimized .tflite model on LFW
#python3 src/main.py --action validate_tflite --model_dir log/tflite/quantized_model.tflite

# Transform the .pb model to a .tflite optimized (the .tflite model will be written at ./log/tflite/)
#python3 src/main.py --action convert_model --model_dir log/frozen_model


# Transforme the checkpoint model in a .pb model
#python3 tensorflow/tensorflow/python/tools/freeze_graph.py --input_meta_graph=DocFace-master/log/faceres_ms/graph.meta --input_checkpoint=DocFace-master/log/faceres_ms/ckpt-320000 --output_graph=DocFace-master/log/model/frozen_graph.pb --output_node_names=outputs --input_binary=True


# extract features for some images described in './result/images.txt' and put those features at './result/output.py'
#python src/extract_features.py --model_dir ./log/faceres_ms --image_list ./result/images.txt --output ./result/output.npy

