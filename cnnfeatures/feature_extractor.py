
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications import inception_resnet_v2, inception_v3, resnet50, xception, vgg16, vgg19, mobilenet, nasnet
from keras.models import Model
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K
import keras

class FeatureExtractor:

    def __init__(self, model, input_size):
        
        input_shape = (input_size, input_size, 3)
        
        if model == 'xception':
            base_model = xception.Xception(weights='imagenet', include_top=False, pooling='max', input_shape=input_shape)
        elif model == 'vgg16':
            base_model = vgg16.VGG16(weights='imagenet', include_top=False, pooling='max', input_shape=input_shape)
        elif model == 'vgg19':
            base_model = vgg19.VGG19(weights='imagenet', include_top=False, pooling='max', input_shape=input_shape)
        elif model == 'inception_v3':
            base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='max', input_shape=input_shape)
        elif model == 'mobilenet':
            base_model = mobilenet.MobileNet(weights='imagenet', include_top=False, pooling='max', input_shape=input_shape)
        elif model == 'inception_resnet_v2':
            base_model = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, pooling='max', input_shape=input_shape)
        elif model == 'resnet50':
            base_model = resnet50.ResNet50(weights='imagenet', include_top=False, pooling='max', input_shape=input_shape)
        elif model == 'nasnetlarge':
            base_model = nasnet.NASNetLarge(weights='imagenet', include_top=False, pooling='max', input_shape=input_shape)
        else:
            base_model = nasnet.NASNetMobile(weights='imagenet', include_top=False, pooling='max', input_shape=input_shape)
            
        self.input_size = input_size
        self.model = base_model
        self.graph = tf.get_default_graph()
        base_model.summary()

    def extract(self, img):  # img is from PIL.Image.open(path) or keras.preprocessing.image.load_img(path)

        img = img.resize((self.input_size, self.input_size)) 
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel

        with self.graph.as_default():
            feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
            # print(feature)
            # newModel = K.function([self.model.layers[0].input], [self.model.layers[-2].output, self.model.layers[-1].output])
            # newModel = K.function([self.model.layers[0].input], [self.model.get_layer('block5_conv4').output,
            #                                                      self.model.get_layer('global_max_pooling2d_1').output])
            # f1, f2 = newModel([x])
            # print(np.shape(f1[0]))
            # print(np.shape(f2[0]))
            return feature / np.linalg.norm(feature)  # Normalize


if __name__ == '__main__':
    img = Image.open('/media/chenlei/DATA/test/581.png')
    fe = FeatureExtractor('vgg19', 224)
    # fe.extract(img)
