from backbone import *
from keras.layers import TimeDistributed

def matmul(ip):
    out = tf.matmul(ip[0], ip[1])
    return out

def read_mem(x_query, x_memory, embed_channel, feat_channel, nb_classes, nb_head):

    x_query_reshape = Reshape((1, feat_channel), input_shape=(feat_channel,))(x_query)
    q = TimeDistributed(Dense((embed_channel*nb_head), activation='linear'))(x_query_reshape)
    q = Reshape((1, nb_head, embed_channel), input_shape=(1, (nb_head*embed_channel)))(q)
    q = Permute((2, 1, 3))(q)
    k = TimeDistributed(Dense((embed_channel*nb_head), activation='linear'))(x_memory)
    k = Reshape((nb_classes, nb_head, embed_channel), input_shape=(nb_classes, (nb_head*embed_channel)))(k)
    k = Permute((2, 1, 3))(k)
    v = TimeDistributed(Dense((embed_channel*nb_head), activation='linear'))(x_memory)
    v = Reshape((nb_classes, nb_head, embed_channel), input_shape=(nb_classes, (nb_head*embed_channel)))(v)
    v = Permute((2, 1, 3))(v)
    k = Permute((1, 3, 2))(k)
    r = Lambda(matmul)([q, k])
    r = Lambda(lambda x: x/(embed_channel**0.5))(r)
    r = Activation('softmax')(r)
    z = Lambda(matmul)([r, v])
    z = Permute((2, 1, 3))(z)
    z = Reshape((1, (nb_head*embed_channel)), input_shape=(1, nb_head, embed_channel))(z)
    z = Flatten()(z)
    return z

def pmnet(backbone='resnet50', nb_classes=20, patch_size=224, feat_channel=512, embed_channel=1024, trainable=False, nb_head=20):

    if backbone=='resnet50':
        base_model = resnet50(nb_classes, patch_size, None, None, True)
        feat_channel = 2048
    
    if backbone=='nasnet':
        base_model = nasnet(nb_classes, patch_size, None, None, True)
        feat_channel = 1056

    if backbone == 'vggnet16':
        base_model = vggnet16(nb_classes, patch_size, None, None, True)
        feat_channel = 512

    if backbone == 'inceptionv3':
        base_model = inceptionv3(nb_classes, patch_size, None, None, True)
        feat_channel = 2048

    for lyr in range(len(base_model.layers)):
        base_model.layers[lyr].trainable = trainable

    x_memory = Input(shape=(nb_classes, feat_channel), name='memory')
    x_query = base_model.output

    x = read_mem(x_query, x_memory, embed_channel, feat_channel, nb_classes, nb_head)

    x = Dense(nb_classes)(x)
    x_out = Activation('sigmoid')(x)

    model = Model([base_model.input, x_memory], x_out, name='pmnet')
    return model


