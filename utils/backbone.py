from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Activation, UpSampling2D, ZeroPadding2D, Flatten, TimeDistributed, Reshape, Bidirectional, Permute, Lambda, Add, Concatenate, Dot, Dropout, Multiply, RepeatVector, LSTM
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.pooling import GlobalMaxPooling2D
from keras import regularizers, layers
from keras import backend as K
from keras.initializers import Constant
import tensorflow as tf
from keras.applications.resnet import ResNet50 #ResNet152, ResNet101, ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.densenet import DenseNet121, DenseNet201, DenseNet169
from keras.applications.inception_v3 import InceptionV3
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.optimizers import SGD, RMSprop, Nadam
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

print('tensorflow version:', tf.__version__)

def inceptionv3(nb_classes=20, patch_size=224, activation='softmax', pretrain='imagenet', isbackbone=False):

    base_model = InceptionV3(include_top=False, weights=pretrain, input_tensor=Input(shape=(patch_size, patch_size, 3)), input_shape=(patch_size, patch_size, 3), pooling='avg')

    x = base_model.output
    if isbackbone:
        return Model(base_model.input, base_model.output, name='inceptionv3')
    else:
        x = Dense(nb_classes, activation=activation, name='classify')(x)
        return Model(base_model.input, x, name='inceptionv3')

def resnet50(nb_classes=20, patch_size=224, activation='softmax', pretrain='imagenet', isbackbone=False):

    base_model = ResNet50(include_top=False, weights=pretrain, input_tensor=Input(shape=(patch_size, patch_size, 3)), input_shape=(patch_size, patch_size, 3), pooling='avg')
    x = base_model.output
    if isbackbone:
        return Model(base_model.input, base_model.output, name='resnet50')
    else:
        x = Dense(nb_classes, activation=activation, name='classify')(x)
        return Model(base_model.input, x, name='resnet50')

def nasnet(nb_classes=20, patch_size=224, activation='softmax', pretrain='imagenet', isbackbone=False):

    base_model = NASNetMobile(include_top=False, weights=pretrain, input_tensor=Input(shape=(patch_size, patch_size, 3)), input_shape=(patch_size, patch_size, 3), pooling='avg')

    x = base_model.output
    if isbackbone:
        return Model(base_model.input, base_model.output, name='nasnet')
    else:
        x = Dense(nb_classes, activation=activation, name='classify')(x)
        return Model(base_model.input, x, name='nasnet')

def vggnet16(nb_classes=20, patch_size=224, activation='softmax', pretrain='imagenet', isbackbone=False):

    base_model = VGG16(include_top=False, weights=pretrain, input_tensor=Input(shape=(patch_size, patch_size, 3)), input_shape=(patch_size, patch_size, 3), pooling='avg')
   
    x = base_model.get_layer('block5_pool').output
    #x = BatchNormalization()(x)
    if isbackbone:
        x = GlobalAveragePooling2D()(x)
        return Model(base_model.input, x, name='vggnet16')
    else:
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(nb_classes, activation=activation, name='classify')(x)
        return Model(base_model.input, x, name='vggnet16')




