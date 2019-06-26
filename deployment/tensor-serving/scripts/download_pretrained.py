#Download and save pretrained models as keras models *.h5
from keras.applications import vgg16, inception_v3, resnet50, mobilenet

#path_to_save = "../../../models/classification/images/pretrained/keras/"
path_to_save = "models/classification/images/pretrained/keras/"

#Load and save the VGG model as *.h5 file
vgg_model = vgg16.VGG16(weights='imagenet')
vgg_model.save(''.join([path_to_save,'vgg.h5']))

#Load and save the Inception_V3 model as *.h5 file
inception_model = inception_v3.InceptionV3(weights='imagenet')
inception_model.save(''.join([path_to_save,'inception.h5']))

#Load and save the ResNet50 model as *.h5 file
resnet_model = resnet50.ResNet50(weights='imagenet')
resnet_model.save(''.join([path_to_save,'resnet.h5']))

#Load and save the MobileNet model as *.h5 file
mobilenet_model = mobilenet.MobileNet(weights='imagenet')
mobilenet_model.save(''.join([path_to_save,'mobilenet.h5']))
