import numpy as np
import argparse
from basenet import BaseNetwork
import utils
from lfw import LFWTest

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
def main(args):

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

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_dir", help="The path to the pre-trained model directory", type=str, default= './log/faceres_ms')
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


  args = parser.parse_args()
  main(args)


#python src/extract_features.py --model_dir ./log/faceres_ms --image_list ./result/images.txt --output ./result/output.npy

