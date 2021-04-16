from backbone import *

def cnn(backbone, nb_classes=20, patch_size=224, activation='softmax', pretrain='imagenet', trainable=False, isbackbone=False):
    if backbone=='resnet50':
        base_model = resnet50(nb_classes, patch_size, activation, pretrain, isbackbone)
        feat_channel = 2048

    if backbone=='nasnet':
        base_model = nasnet(nb_classes, patch_size, activation, pretrain, isbackbone)
        feat_channel = 1056

    if backbone == 'vggnet16':
        base_model = vggnet16(nb_classes, patch_size, activation, pretrain, isbackbone)
        feat_channel = 512

    if backbone == 'inceptionv3':
        base_model = inceptionv3(nb_classes, patch_size, activation, pretrain, isbackbone)
        feat_channel = 2048

    for lyr in range(len(base_model.layers)):
        base_model.layers[lyr].trainable = trainable

    return Model(base_model.input, base_model.output, name='cnn')

def sel_dim(backbone):

    if backbone=='resnet50':
        feat_channel = 2048

    if backbone=='nasnet':
        feat_channel = 1056

    if backbone == 'vggnet16':
        feat_channel = 512

    if backbone == 'inceptionv3':
        feat_channel = 2048

    return feat_channel
