# Introduction

### auto select best classifier 
classifiers.py: Classifier creator functions based on sklearn 
select_classifier.py: based on classifiers.py, create classifier list to auto select best classifier.

### duplicate image checking 
tools.py: some common functions for image processing
feature_detection.py: use SIFT/SURF/ORB to extract image features for duplicate image checking
hist_detection.py: use color distribution for duplicate image checking
dHash.py: calculate images hamming distance for duplicate image checking

### image info extraction(OCR) 
tesseract_image.py: use tesseract to extract text from images for document classification

### CNN
vgg16.py: VGG16 for tensorflow
