# Introduction

* tools.py: some common functions for image processing
* dbhelper.py: database related functions

### machine_learning(auto select best classifier) 
* classifiers.py: Classifier creator functions based on sklearn, include Bayes,KNN,LR,Random Forest,Decision Tree,GBDT,SVM...
* select_classifier.py: based on classifiers.py, create classifier list to auto select best classifier.

### feature_extraction(duplicate image checking) 

* feature_detection.py: use SIFT/SURF/ORB to extract image features for duplicate image checking
* hist_detection.py: use color distribution for duplicate image checking
* d_hash.py: calculate images hamming distance for duplicate image checking

### ocr(image info extraction, NLP) 
* tesseract_image.py: use tesseract/nltk to extract/analyze text from images for document classification

### deep_learning(CNN models for fine-tune)
* vgg16_tensorflow.py: VGG16 for tensorflow
* vggKeras.py.py: VGG16/19 for keras
* cnn.py: integrate VGG16/VGG19/InceptionV3/Xception/ResNet50 cnn selector for keras, TODO:: add finetune based on vgg16-keras.py

### image_crawler(image collection from web)
* icrawler_test.py: using icrawler to collect images from google, baidu, yahoo
* merge_image.py: check all images & convert them into png
