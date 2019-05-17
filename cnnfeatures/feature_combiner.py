
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications import inception_resnet_v2, inception_v3, resnet50, xception, vgg16, vgg19, mobilenet, nasnet
from keras.models import Model
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K

class FeatureCombiner:

    def __init__(self, input_size=224):
        
        input_shape = (input_size, input_size, 3)

        xception_model = xception.Xception(weights='imagenet', include_top=False, pooling='max', input_shape=input_shape)
        inceptionv3_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='max', input_shape=input_shape)
        resnet50_model = resnet50.ResNet50(weights='imagenet', include_top=False, pooling='max', input_shape=input_shape)
        vgg19_model = vgg19.VGG19(weights='imagenet', include_top=False, pooling='max', input_shape=input_shape)

        self.input_size = input_size
        self.models = [xception_model,inceptionv3_model,resnet50_model,vgg19_model]
        self.graph = tf.get_default_graph()

    def extract(self, img):  # img is from PIL.Image.open(path) or keras.preprocessing.image.load_img(path)

        img = img.resize((self.input_size, self.input_size)) 
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel

        with self.graph.as_default():
            output = []
            for model in self.models:
                feature = model.predict(x)[0]  # (1, N) -> (N, )
                output.extend(feature)
            return output / np.linalg.norm(output)  # Normalize


if __name__ == '__main__':

    img = Image.open('/media/chenlei/DATA/test/test.png')
    fe = FeatureCombiner()
    fe.extract(img)
