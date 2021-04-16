import numpy as np
import scipy.io as sio
import h5py
import random
from PIL import Image
import tensorflow as tf
import keras.backend as K
import os
import glob

def gpu_config(gpu_id=0, gpu_mem=0.8):
    if gpu_mem:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_mem)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        print('############## gpu memory is fixed to', gpu_mem, 'on', str(gpu_id), '#################')
    else:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        print('gpu memory is flexible.')

folder_path = 'mai/'

def load_data(config='ucm2mai', im_size=224, evaluate=True, tr_test=0.8):

    if config=='ucm_si' or config=='ucm_mu' or config=='aid_si' or config=='aid_mu':
        return SetData_single(config, im_size, tr_test)

    X_trainval = []
    y_trainval = []
    X_test = []
    y_test = []

    # get trainval and test split
    config_trainval = '{}configs/{}_trainval.txt'.format(folder_path, config)
    config_test = '{}configs/{}_test.txt'.format(folder_path, config)
    f = sio.loadmat('{}multilabel.mat'.format(folder_path))
    labels = f['labels']
    
    if config=='aid2mai':
        temp = labels
        labels = np.zeros((len(temp), 20))
        labels[:, 0] = temp[:, 0]
        labels[:, 1] = temp[:, 1]
        labels[:, 2] = temp[:, 2]
        labels[:, 3] = temp[:, 16]
        labels[:, 4] = temp[:, 3]
        labels[:, 5] = temp[:, 4]
        labels[:, 6] = temp[:, 5]
        labels[:, 7] = temp[:, 17]
        labels[:, 8] = temp[:, 6]
        labels[:, 9] = temp[:, 18]
        labels[:, 10] = temp[:, 7]
        labels[:, 11] = temp[:, 8]
        labels[:, 12] = temp[:, 9]
        labels[:, 13] = temp[:, 19]
        labels[:, 14] = temp[:, 20]
        labels[:, 15] = temp[:, 21]
        labels[:, 16] = temp[:, 10]
        labels[:, 17] = temp[:, 22]
        labels[:, 18] = temp[:, 23]
        labels[:, 19] = temp[:, 11]
   
    if config=='ucm2mai':
        temp = labels
        labels = np.zeros((len(temp), 16))
        labels[:, 0] = temp[:, 0]
        labels[:, 1] = temp[:, 1]
        labels[:, 2] = temp[:, 2]
        labels[:, 3] = temp[:, 3]
        labels[:, 4] = temp[:, 4]
        labels[:, 5] = temp[:, 5]
        labels[:, 6] = temp[:, 12] 
        labels[:, 7] = temp[:, 6]
        labels[:, 8] = temp[:, 7]
        labels[:, 9] = temp[:, 8]
        labels[:, 10] = temp[:, 9]
        labels[:, 11] = temp[:, 13]
        labels[:, 12] = temp[:, 14]
        labels[:, 13] = temp[:, 10]
        labels[:, 14] = temp[:, 15]
        labels[:, 15] = temp[:, 11]


    # load all test images
    with open(config_test) as f:
        test_list = f.readlines()

    for i in range(len(test_list)):
        fn_id = int(test_list[i][:-1])
        im = np.float32(Image.open('{}images/{}.jpg'.format(folder_path, str(fn_id))).resize((im_size, im_size)))
        im = im[:, :, [2, 1, 0]] # PIL to cv2
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
        label = labels[fn_id-1, :]
        X_test.append(im)
        y_test.append(label)

    X_test = np.float32(X_test)
    y_test = np.uint8(y_test)

    if evaluate:
        return X_trainval, y_trainval, X_test, y_test

    # load all training images
    with open(config_trainval) as f:
        trainval_list = f.readlines()

    for i in range(len(trainval_list)):
        fn_id = int(trainval_list[i][:-1])
        im = np.float32(Image.open('{}images/{}.jpg'.format(folder_path, str(fn_id))).resize((im_size, im_size)))
        im = im[:, :, [2, 1, 0]] # PIL to cv2
        #im = cv2.resize(cv2.imread('{}images/{}.jpg'.format(folder_path, str(fn_id))), (im_size, im_size)).astype(np.float32)
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
        label = labels[fn_id-1, :]
        X_trainval.append(im)
        y_trainval.append(label)

    X_trainval = np.float32(X_trainval)
    y_trainval = np.uint8(y_trainval)

    return X_trainval, y_trainval, X_test, y_test


def SetData_single(dataset, im_size, tr_test=0.8):

    X = []
    y = []
    num_cls = []
    ind = 0

    # prepare images
    if dataset == 'ucm_mu' or dataset== 'ucm_si':

        im_path = '/data/data/UCM_dataset/'
        #label_path = folder_path + 'UCM_dataset/multilabel.mat'

        classes = ['airplane', 'baseballdiamond', 'beach', 'buildings', 'agricultural','forest', 'golfcourse', 'parkinglot', 'harbor', 'denseresidential', 'river', 'runway', 'chaparral', 'storagetanks', 'tenniscourt', 'mediumresidential']
        nb_classes_single = len(classes)
        nb_classes_multi = 16

        for cls_id in range(nb_classes_single):
            print(cls_id)
            cls = classes[cls_id]
            nb_samples = 100#len(glob.glob(im_path+'{}/{}*'.format(cls, str(cls).lower())))
            num_cls.append(nb_samples)
            #num_sample = 0# in case of umbalanced samples

            for fn_id in range(100):
                im = np.float32(Image.open(im_path+'{}/{}.tif'.format(cls, cls+str(fn_id).zfill(2))).resize((im_size, im_size)))
                im = im[:, :, [2, 1, 0]]
                im[:,:,0] -= 103.939
                im[:,:,1] -= 116.779
                im[:,:,2] -= 123.68
                X.append(im)
                if dataset == 'ucm_mu':
                    label = np.zeros((nb_classes_multi))
                    if cls == 'beach':
                        label[2] = 1
                        label[15] = 1
                    elif cls == 'mediumresidential':
                        label[9] = 1
                    else:
                        label[cls_id] = 1

                elif dataset == 'ucm_si':
                    label = np.zeros((nb_classes_single))
                    label[cls_id] = 1

                if fn_id==0:
                    print(cls, label)
                y.append(label)
                ind += 1

            #num_cls.append(num_sample)

    elif dataset == 'aid_mu' or dataset == 'aid_si':

        im_path = '/data/data/AID_dataset/'
        #label_path = folder_path + 'AID_dataset/multilabel.mat'
        classes = ['Airport', 'BaseballField', 'Beach', 'Bridge', 'Commercial', 'Farmland', 'Forest', 'Pond', 'Parking', 'Park', 'Port', 'DenseResidential', 'River', 'Viaduct', 'Playground', 'Stadium', 'StorageTanks', 'RailwayStation', 'Industrial', 'MediumResidential']

        nb_classes_single = len(classes)
        nb_classes_multi = 20

        for cls_id in range(nb_classes_single):
            cls = classes[cls_id]
            nb_samples = len(glob.glob(im_path+'{}/{}*'.format(cls, str(cls).lower())))
            #print(im_path+'{}/{}*'.format(cls, str(cls).lower()))
            #print(cls_id, cls, nb_samples)
            print('\''+cls+'\', ')
            num_cls.append(nb_samples)
            #num_sample = 0# in case of umbalanced samples

            for fn_id in range(nb_samples):
                im = np.float32(Image.open(im_path+'{}/{}.jpg'.format(cls, str(cls).lower() + '_' +str(fn_id+1))).resize((im_size, im_size)))
                im = im[:, :, [2, 1, 0]]
                im[:,:,0] -= 103.939
                im[:,:,1] -= 116.779
                im[:,:,2] -= 123.68
                X.append(im)
                if dataset == 'aid_mu':
                    label = np.zeros((nb_classes_multi))
                    if cls == 'Beach':
                        label[2] = 1
                        label[19] = 1
                    elif cls == 'MediumResidential':
                        label[11] = 1
                    else:
                        label[cls_id] = 1

                elif dataset == 'aid_si':
                    label = np.zeros((nb_classes_single))
                    label[cls_id] = 1

                #if fn_id==0:
                    #print(cls, label)

                y.append(label)
                ind += 1

    # format
    X = np.float32(X)
    y = np.uint8(y)
    print(np.shape(X))
    print(np.shape(y))
    print(np.sum(y, 0))

    # random and find out test, tr

    num_samples = range(sum(num_cls))
    train_ix, tr_ix = [], []
    test_ix, te_ix = [], []

    ind = 0
    # every class shcls in range(nb_classes):
    for cls in range(nb_classes_single):
        samp = num_samples[ind:(ind+num_cls[cls])]
        tr, test = np.split(samp, [int(num_cls[cls]*tr_test)])
        tr_ix.append(tr)
        te_ix.append(test)
        ind += num_cls[cls]

    for i in range(len(tr_ix)):
        for j in range(len(tr_ix[i])):
            train_ix.append(tr_ix[i][j])

    for i in range(len(te_ix)):
        for j in range(len(te_ix[i])):
            test_ix.append(te_ix[i][j])

    # random.seed(75814)
    # random.shuffle(train_ix)
    # split dataset to train and test
    X_tr = [X[int(i)] for i in train_ix]
    y_tr = [y[int(i)] for i in train_ix]

    X_tr = np.float32(X_tr)
    y_tr = np.uint8(y_tr)

    X_test = [X[int(i)] for i in test_ix]
    y_test = [y[int(i)] for i in test_ix]

    X_test = np.float32(X_test)
    y_test = np.uint8(y_test)

    del X, y

    return X_tr, y_tr, X_test, y_test


'''
def SetData_multi(dataset, im_size, one_shot=False):

    im = []
    X_tr = []
    y_tr = []
    X_test = []
    y_test = []
    num_cls = []

    # prepare images
    if dataset == 'ucm_si2mu':

        im_path = folder_path + 'UCM_si2mu/'
        label_path = folder_path + 'UCM_si2mu/multilabel.mat'

    elif dataset == 'aid_si2mu':
       
        im_path = folder_path + 'AID_si2mu/'
        label_path = folder_path + 'AID_si2mu/multilabel.mat'

    labels_mat = sio.loadmat(label_path)
    labels = labels_mat['labels']
    nb_classes = len(labels[0])

    if not one_shot:
        trainval_folder = 'images_trainval_more/'
        test_folder = 'images_test_more/'
    else:
        trainval_folder = 'images_trainval/'
        test_folder = 'images_test/'

    filenames_trainval = os.listdir(im_path+trainval_folder)
    for i in range(len(filenames_trainval)):
        #print(i)
        im = cv2.resize(cv2.imread(im_path+trainval_folder+filenames_trainval[i]), (im_size, im_size)).astype(np.float32)
        im_copy = im.copy()
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        X_tr.append(im)
        y_tr.append(labels[int(filenames_trainval[i][:-4])-1, :])

    filenames_test = os.listdir(im_path+test_folder)
    print(np.shape(filenames_test))
    for i in range(len(filenames_test)):
        #print(i)
        im = cv2.resize(cv2.imread(im_path+test_folder+filenames_test[i]), (im_size, im_size)).astype(np.float32)
        im_copy = im.copy()
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        X_test.append(im)
        y_test.append(labels[int(filenames_test[i][:-4])-1, :])

    # format
    X_tr = np.float32(X_tr)
    y_tr = np.uint8(y_tr)
    X_test = np.float32(X_test)
    y_test = np.uint8(y_test)

    return X_tr, y_tr, X_test, y_test
'''
def SetData_si2mu_special(dataset, im_size, tr_test=0.8):

    im = []
    X = []
    y = []
    X_tr = []
    y_tr = []
    X_test = []
    y_test = []

    # prepare images
    if dataset == 'ucm_mul':

        im_path = folder_path + 'UCM_dataset/'
        label_path = folder_path + 'UCM_dataset/multilabel.mat'

        classes = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt']
        nb_classes = len(classes)

        for cls in classes:
            num_sample = 0# in case of umbalanced samples

            for fn_id in range(100):
                im = cv2.resize(cv2.imread(im_path+'{}/{}.tif'.format(cls, cls+str(fn_id).zfill(2))), (im_size, im_size)).astype(np.float32)
                im_copy = im.copy()
                im[:,:,0] -= 103.939
                im[:,:,1] -= 116.779
                im[:,:,2] -= 123.68
                X.append(im)
                X_ori.append(im_copy)

                num_sample += 1
                ind += 1

            num_cls.append(num_sample)

    elif dataset == 'aid_si2mu':
        #################################################33
        # load multilabel image
        ###################################################
        im_path = folder_path + 'AID_si2mu/'
        label_path = folder_path + 'AID_si2mu/multilabel.mat'
        labels_mat = sio.loadmat(label_path)
        labels = labels_mat['labels']
        nb_classes = len(labels[0])

        filenames_trainval = os.listdir(im_path+'images_trainval_more/')
        for i in range(len(filenames_trainval)):
            im = cv2.resize(cv2.imread(im_path+'images_trainval_more/'+filenames_trainval[i]), (im_size, im_size)).astype(np.float32)
            im_copy = im.copy()
            im[:,:,0] -= 103.939
            im[:,:,1] -= 116.779
            im[:,:,2] -= 123.68
            X_tr.append(im)
            y_tr.append(labels[int(filenames_trainval[i][:-4])-1, :])

        filenames_test = os.listdir(im_path+'images_test_more/')
        print(np.shape(filenames_test))
        for i in range(len(filenames_test)):
            im = cv2.resize(cv2.imread(im_path+'images_test_more/'+filenames_test[i]), (im_size, im_size)).astype(np.float32)
            im_copy = im.copy()
            im[:,:,0] -= 103.939
            im[:,:,1] -= 116.779
            im[:,:,2] -= 123.68
            X_test.append(im)
            y_test.append(labels[int(filenames_test[i][:-4])-1, :])

        # format
        X_tr_mul = np.float32(X_tr)
        y_tr_mul = np.uint8(y_tr)
        X_test = np.float32(X_test)
        y_test = np.uint8(y_test)

        #################################################33
        # load multilabel image
        ###################################################

        im_path = folder_path + 'AID_dataset/'

        classes = ['Airport', 'BaseballField', 'Beach', 'Bridge', 'Commercial', 'Farmland', 'Forest', 'Pond', 'Parking', 'Park', 'Port', 'DenseResidential', 'River', 'Viaduct', 'Playground', 'Stadium', 'StorageTanks', 'RailwayStation', 'Industrial', 'MediumResidential']

        nb_classes_single = len(classes)
        nb_classes_multi = 20

        for cls_id in range(nb_classes_single):
            cls = classes[cls_id]
            nb_samples = len(glob.glob(im_path+'{}/{}*'.format(cls, str(cls).lower())))
            #num_cls.append(nb_samples)
            #num_sample = 0# in case of umbalanced samples
            print(cls)
            for fn_id in range(nb_samples):
                im = cv2.resize(cv2.imread(im_path+'{}/{}.jpg'.format(cls, str(cls).lower() + '_' +str(fn_id+1))), (im_size, im_size)).astype(np.float32)
                im[:,:,0] -= 103.939
                im[:,:,1] -= 116.779
                im[:,:,2] -= 123.68
                X.append(im)
                label = np.zeros((nb_classes_multi))
                if cls == 'Beach':
                    label[2] = 1
                    label[19] = 1
                elif cls == 'MediumResidential':
                    label[11] = 1
                else:
                    label[cls_id] = 1

                y.append(label)
                #ind += 1

        # format
        X_tr_si = np.float32(X)
        y_tr_si = np.uint8(y)
        print(np.shape(X))
        nb_samples_per_class = np.sum(y_tr_si, 0)
        print(nb_samples_per_class)
        #################################################33
        # create training set
        ###################################################
        X_tr = []
        X_tr2 = []
        y_tr2 = []
        idx_range_mul = range(nb_classes_multi)
        for i in range(len(y_test)):
            #print('a set:')
            for j in range(nb_classes_single):
                idx_range_si = np.where(y_tr_si[:,j]==1)[0]
                idx = idx_range_si[int(np.floor(random.random()*len(idx_range_si)))]
                #print(idx)
                im_idx = np.expand_dims(X_tr_si[idx, :, :, :], axis=0)
                im_options = np.concatenate([im_options, im_idx], axis=0) if j>0 else im_idx
                #print(np.shape(im_options))

            X_tr.append(im_options)
            idx_mul = idx_range_mul[int(np.floor(random.random()*len(idx_range_mul)))]
            #print(idx_mul)
            X_tr2.append(X_tr_mul[idx_mul])
            y_tr2.append(y_tr_mul[idx_mul])
        
        X_tr_si = np.float32(X_tr)
        print('X_tr_si:', np.shape(X_tr_si))
        X_tr_mul = np.float32(X_tr2)
        y_tr = np.float32(y_tr2)
        print('X_tr_mul:', np.shape(X_tr_mul))
        ''' 
        sio.savemat('X_tr_si.mat', {'X_tr_si':X_tr_si})
        sio.savemat('X_tr_mul.mat', {'X_tr_mul':X_tr_mul})
        sio.savemat('y_tr.mat', {'y_tr':y_tr})
        print('mat are saved')
        '''

    return X_tr_si, X_tr_mul, y_tr, X_test, y_test




def mAP(y_true, y_pred):
    ignore = 0.5

    pred = K.round(y_pred)
    gt = y_true

    mask_keep = K.cast(~K.equal(y_true, ignore), K.floatx())
    #pred_mask = pred*mask_keep
    #gt_mask = gt*mask_keep

    # remove ignore value
    pg_minus = (pred-gt)
    pg_add = (pred+gt)

    # calculate TP, FP, FN, TN
    TP_l = K.sum(K.cast(K.equal(pg_add, 2), K.floatx()), 0)
    FP_l = K.sum(K.cast(K.equal(pg_minus, 1), K.floatx()), 0)
    FN_l = K.sum(K.cast(K.equal(pg_minus, -1), K.floatx()), 0)
    TN_l = K.sum(K.cast(K.equal(pg_add, 0), K.floatx()), 0)

    # mAP
    AP = TP_l/(TP_l+FP_l+1e-12)
    
    return K.mean(AP)

    

def PR_score(pred, gt):

    pred = np.int8(np.round(pred))
    gt = np.int8(np.round(gt))

    pg_minus = pred-gt
    pg_add = pred+gt

    TP_mat = (pg_add == 2)
    FP_mat = (pg_minus == 1)
    FN_mat = (pg_minus == -1)
    TN_mat = (pg_add == 0)

    # calculate example-based
    TP_e = np.float32(np.sum(TP_mat, 1))
    FP_e = np.float32(np.sum(FP_mat, 1))
    FN_e = np.float32(np.sum(FN_mat, 1))
    TN_e = np.float32(np.sum(TN_mat, 1))

    # clear TP_e is 0, assign it some value and latter assign zero
    ind_zero = np.where(TP_e==0)
    TP_e[ind_zero] = 0.0000001

    precision_e = TP_e/(TP_e+FP_e)
    recall_e = TP_e/(TP_e+FN_e)

    precision_e[ind_zero] = 0
    recall_e[ind_zero] = 0

    # calculate label-based
    TP_l = np.float32(np.sum(TP_mat, 0))
    FP_l = np.float32(np.sum(FP_mat, 0))
    FN_l = np.float32(np.sum(FN_mat, 0))
    TN_l = np.float32(np.sum(TN_mat, 0))

    # clear TP_l is 0, assign it some value and latter assign zero
    ind_zero = np.where(TP_l==0)
    TP_l[ind_zero] = 0.0000001

    precision_l = TP_l/(TP_l+FP_l)
    recall_l = TP_l/(TP_l+FN_l)

    precision_l[ind_zero] = 0
    recall_l[ind_zero] = 0

    return precision_e, recall_e, precision_l, recall_l

def F_score(pred, gt, ex_base=True, beta=1):

    if ex_base:
        precision, recall, _, _ = PR_score(pred, gt)
    else:
        _, _, precision, recall = PR_score(pred, gt)

    print(np.shape(precision), np.shape(recall))
    if precision.any()<0 or recall.any()<0:
        print('negative precision or recall!!!!')
        F = -999999999
        return F
    pr = (beta*beta)*precision+recall
    #check both zeros    
    ind_zero = np.where(pr==0)
    pr[ind_zero] = 0.0000001
    
    F = (1.0+beta*beta)*(precision*recall)/pr
    F[ind_zero] = 0

    return F

def F1_e(y_true, y_pred):

    gt = y_true
    pred = K.round(y_pred)

    pg_minus = pred-gt
    pg_add = pred+gt

    # calculate example-based
    TP_e = K.sum(K.cast(K.equal(pg_add, 2), K.floatx()), 1)
    FP_e = K.sum(K.cast(K.equal(pg_minus, 1), K.floatx()), 1)
    FN_e = K.sum(K.cast(K.equal(pg_minus, -1), K.floatx()), 1)
    TN_e = K.sum(K.cast(K.equal(pg_add, 0), K.floatx()), 1)

    # in case of 0
    TP_e2 = TP_e * K.cast(~K.equal(TP_e, 0), K.floatx()) + K.cast(K.equal(TP_e, 0), K.floatx())
    precision_e = TP_e2/(TP_e2+FP_e)
    recall_e = TP_e2/(TP_e2+FN_e)
    precision_e = precision_e * K.cast(~K.equal(TP_e, 0), K.floatx())
    recall_e = recall_e * K.cast(~K.equal(TP_e, 0), K.floatx())

    pr = precision_e+recall_e
    pr2 = pr * K.cast(~K.equal(pr, 0), K.floatx()) + K.cast(K.equal(pr, 0), K.floatx())
    F = 2*(precision_e*recall_e)/pr2
    F = F * K.cast(~K.equal(pr, 0), K.floatx())
    
    return K.mean(F)


def F2_e(y_true, y_pred):

    gt = y_true
    pred = K.round(y_pred)

    pg_minus = pred-gt
    pg_add = pred+gt

    # calculate example-based
    TP_e = K.sum(K.cast(K.equal(pg_add, 2), K.floatx()), 1)
    FP_e = K.sum(K.cast(K.equal(pg_minus, 1), K.floatx()), 1)
    FN_e = K.sum(K.cast(K.equal(pg_minus, -1), K.floatx()), 1)
    TN_e = K.sum(K.cast(K.equal(pg_add, 0), K.floatx()), 1)

    # in case of 0
    TP_e2 = TP_e * K.cast(~K.equal(TP_e, 0), K.floatx()) + K.cast(K.equal(TP_e, 0), K.floatx()) # 0 will be replaced by 1, and latter change back to 0
    precision_e = TP_e2/(TP_e2+FP_e)
    recall_e = TP_e2/(TP_e2+FN_e)
    precision_e = precision_e * K.cast(~K.equal(TP_e, 0), K.floatx())
    recall_e = recall_e * K.cast(~K.equal(TP_e, 0), K.floatx())

    pr = 4*precision_e+recall_e
    pr2 = pr * K.cast(~K.equal(pr, 0), K.floatx()) + K.cast(K.equal(pr, 0), K.floatx())
    F = 5*(precision_e*recall_e)/pr2
    F = F * K.cast(~K.equal(pr, 0), K.floatx())
    
    return K.mean(F)


def F12_e(y_true, y_pred):

    gt = y_true
    pred = K.round(y_pred)

    pg_minus = pred-gt
    pg_add = pred+gt

    # calculate example-based
    TP_e = K.sum(K.cast(K.equal(pg_add, 2), K.floatx()), 1)
    FP_e = K.sum(K.cast(K.equal(pg_minus, 1), K.floatx()), 1)
    FN_e = K.sum(K.cast(K.equal(pg_minus, -1), K.floatx()), 1)
    TN_e = K.sum(K.cast(K.equal(pg_add, 0), K.floatx()), 1)

    # in case of 0
    TP_e2 = TP_e * K.cast(~K.equal(TP_e, 0), K.floatx()) + K.cast(K.equal(TP_e, 0), K.floatx())
    precision_e = TP_e2/(TP_e2+FP_e)
    recall_e = TP_e2/(TP_e2+FN_e)
    precision_e = precision_e * K.cast(~K.equal(TP_e, 0), K.floatx())
    recall_e = recall_e * K.cast(~K.equal(TP_e, 0), K.floatx())

    pr = precision_e+recall_e
    pr2 = pr * K.cast(~K.equal(pr, 0), K.floatx()) + K.cast(K.equal(pr, 0), K.floatx())
    F1 = 2*(precision_e*recall_e)/pr2
    F1 = F1 * K.cast(~K.equal(pr, 0), K.floatx())

    pr = 4*precision_e+recall_e
    pr2 = pr * K.cast(~K.equal(pr, 0), K.floatx()) + K.cast(K.equal(pr, 0), K.floatx())
    F2 = 5*(precision_e*recall_e)/pr2
    F2 = F2 * K.cast(~K.equal(pr, 0), K.floatx())

    F = 0.7*F1 + 0.3*F2

    return K.mean(F)

