# -*- coding: UTF-8 -*-

# import the necessary packages
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception  # tensorflow only
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils  # 模块中有一些函数可以方便的进行输入图像预处理和解码输出分类
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
# import cv2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def decode_predictions_custom(preds, top=5):
    CLASS_CUSTOM = ["0","1","2","3","4","5","6","7","8","9"]
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_CUSTOM[i]) + (pred[i]*100,) for i in top_indices]
        results.append(result)
    return results


if __name__ == '__main__':

    # keras.backend.image_data_format() # 获取当前的维度顺序，默认顺序是 channels_last
    # 解析命令行参数
    # construct the argument(论据) parse and parse(解析) the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument('-i', '--image', default='./image/DSC_0001.JPG',
                    help='path to input image')  # '--image',是要分类的输入图片的路径
    # ap.add_argument('/Users/qyk/Desktop/DSC_0001.JPG')
    ap.add_argument('-model', '--model', type=str, default='vgg16',
                    help='name of pre-trained network to use')  # '--model'，指定想要使用的与训练模型
    args = vars(ap.parse_args())  # 返回对象object的属性和属性值的字典对象
    print(args)

    args['image']='/home/chenlei/clinic1.png'
    print(args)

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    # define a dictionary that maps model names to their classes inside keras
    MODELS = {
        'vgg16': VGG16,
        'vgg19': VGG19,
        'inception': InceptionV3,
        'xception': Xception,
        'resnet': ResNet50
    }

    # ensure a valid model name was supplied via command line argment
    if args['model'] not in MODELS.keys():
        raise AssertionError("The --model command line argument should be a key in the 'MODELS' dictionary")

    # initialize the input image shape (224X224 pixels) along with the pre-processing function (this might need to be changed
    # based on which model we use to classify our image)
    # 经典的CNN输入图像的尺寸，是224×224、227×227、256×256和299×299，但也可以是其他尺寸。
    # VGG16，VGG19和ResNet均接受224×224输入图像，而Inception V3和Xception需要299×299像素输入
    inputShape = (224, 224)
    preprocess = imagenet_utils.preprocess_input

    # if we are using the InceptionV3 or Xception networks, then we
    # need to set the input shape to (299x299) [rather than (224x224)]
    # and use a different image processing function
    if args["model"] in ("inception", "xception"):
        inputShape = (299, 299)
        preprocess = preprocess_input

    # 从磁盘加载预训练的模型weight(权重)并实例化模型
    # load our the network weights from disk (NOTE: if this is the
    # first time you are running this script for a given network, the
    # weights will need to be downloaded first -- depending on which
    # network you are using, the weights can be 90-575MB, so be
    # patient; the weights will be cached and subsequent runs of this
    # script will be *much* faster)
    print("[INFO] loading {}...".format(args["model"]))

    Network = MODELS[args["model"]]  # 从--model命令行参数得到model的名字，通过MODELS词典映射到相应的类
    model = Network(weights="imagenet")  # 然后使用预训练的ImageNet权重实例化卷积神经网络

    model.summary()
    # load the input image using the Keras helper utility while ensuring
    # the image is resized to `inputShape`, the required input dimensions
    # for the ImageNet pre-trained network
    print("[INFO] loading and pre-processing image...")

    image = load_img(args["image"], target_size=inputShape)  # 从磁盘加载输入图像，inputShape调整图像的宽度和高度
    image = img_to_array(image)  # 将图像从PIL/Pillow实例转换为NumPy数组,输入图像现在表示为(inputShape[0],inputShape[1],3)的NumPy数组

    # our input image is now represented as a NumPy array of shape
    # (inputShape[0], inputShape[1], 3) however we need to expand the
    # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
    # so we can pass it through thenetwork
    image = np.expand_dims(image, axis=0)  # 向矩阵添加一个额外的维度(颜色通道),形状(1,inputShape[0],inputShape[1],3)

    # pre-process the image using the appropriate function based on the
    # model that has been loaded (i.e., mean subtraction, scaling, etc.)
    # keras中preprocess_input()函数的作用是对样本执行 逐样本均值消减 的归一化，即在每个维度上减去样本的均值，对于维度顺序是channels_last的数据
    # example:
    # x[..., 0] -= 103.939
    # x[..., 1] -= 116.779
    # x[..., 2] -= 123.68
    image = preprocess(image, mode='tf')  # 调用相应的预处理功能来执行数据归一化

    print(image.shape)
    # 调用CNN中.predict得到预测结果。根据这些预测结果，将它们传递给ImageNet辅助函数decode_predictions，
    # 会得到ImageNet类标签名字(id转换成名字，可读性高)以及与标签相对应的概率
    # classify the image
    print("[INFO] classifying image with '{}'...".format(args["model"]))

    preds = model.predict(image)

    print(preds.shape)

    P = imagenet_utils.decode_predictions(preds)

    # loop over the predictions and display the rank-5 predictions +
    # probabilities to our terminal
    for (i, (imagenetID, label, prob)) in enumerate(P[0]):
        print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

    # load the image via OpenCV, draw the top prediction on the image,
    # and display the image to our screen
    # orig = cv2.imread(args["image"])
    #
    # (imagenetID, label, prob) = P[0][0]
    #
    # cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
    #
    #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    #
    # cv2.imshow("Classification", orig)
    #
    # cv2.waitKey(0)
    #
