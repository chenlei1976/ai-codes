# Introduction

### auto select best classifier 
*1 classifiers.py: Classifier creator functions based on sklearn 

*2 select_classifier.py: based on classifiers.py, create classifier list to auto select best classifier.

### duplicate image checking 
tools.py: some common functions for image processing
feature_detection.py: use SIFT/SURF/ORB to extract image features for duplicate image checking
hist_detection.py: use color distribution for duplicate image checking
dHash.py: calculate images hamming distance for duplicate image checking

### image info extraction(OCR) 
tesseract_image.py: use tesseract to extract text from images for document classification

### CNN
vgg16.py: VGG16 for tensorflow
vgg16-keras.py: VGG16 for keras
cnn.py: integrate VGG16/VGG19/InceptionV3/Xception/ResNet50 cnn selector for keras, TODO:: add finetune based on vgg16-keras.py
