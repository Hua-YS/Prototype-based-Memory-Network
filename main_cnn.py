import os
import sys
import time
sys.path.append(os.path.abspath('./utils'))
import argparse
from mai import *
from cnn import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

parser = argparse.ArgumentParser(description='si2mu')
parser.add_argument('--backbone', default='resnet50', metavar='MODEL', help='vgg16, inceptionv3, resnet50, nasnet (default: resnet50)')
parser.add_argument('--weight_path', default='weights/weights.h5')
parser.add_argument('--pretrain_weight_path', default='weights/aid2mai/resnet50-aid.h5')
parser.add_argument('--data_config', default='aid2mai', metavar='DATA', help='ucm2mai, aid2mai, ucm_si, aid_si (default: aid2mai)')
parser.add_argument('--patch_size', default=224, type=int, metavar='N', help='image size (default: 224)')
parser.add_argument('--lr', default=2e-4, type=float, metavar='LR', help='learning rate (default: 2e-4)')
parser.add_argument('--bs', default=20, type=int, metavar='BS', help='batch size (default: 20)')
parser.add_argument('--ep', default=100, type=int, metavar='EP', help='training epochs (default: 100)')
parser.add_argument('--nonfrozen', default=1, type=int, help='is backbone trainable (default: True)')
parser.add_argument('--evaluate', default=1, type=int, help='evaluate model (default: True)')
parser.add_argument('--activation', default='sigmoid', help='single- or multi-label classification (default: sigmoid)')


# ************************* Configuretion **************************
gpu_config(0, 0.9) # gpu id, consumption

def main():
    global args
    args = parser.parse_args()
    print(args.evaluate)
    print('==================== Loading data ====================')
    X_tr, y_tr, X_test, y_test = load_data(args.data_config, args.patch_size, args.evaluate)
    print('Data config:', args.data_config)
    print('Training data:', len(X_tr))
    print('Test data:', len(X_test))

    print('==================== Building model ===================')
    model = cnn(backbone=args.backbone, nb_classes=y_test.shape[-1], patch_size=args.patch_size, activation=args.activation, pretrain='imagenet', trainable=args.nonfrozen)
    print('Backbone:', args.backbone)

    if args.evaluate:
        print('==================== Evaluating model ===================')
        model.load_weights(args.weight_path, by_name=True)
        out = model.predict(X_test)
        p_e, r_e, p_l, r_l = PR_score(out, y_test)
        f1 = F_score(out, y_test)
        f2 = F_score(out, y_test, True, 2)
        print('f1 | f2 | pe | re | pl | rl \n {f1:.4f} & {f2:.4f} & {pe:.4f} & {re:.4f} & {pl:.4f} & {rl:.4f}'.format(f1=np.mean(f1), f2=np.mean(f2),pe=np.mean(p_e), re=np.mean(r_e), pl=np.mean(p_l), rl=np.mean(r_l)))
        with open('results.txt', 'a+') as f:
            result = 'weights: {weights:s} \ntime: {time:s} \nresults: f1 | f2 | pe | re | pl | rl \n{f1:.4f} & {f2:.4f} & {pe:.4f} & {re:.4f} & {pl:.4f} & {rl:.4f} \n'.format(weights=args.weight_path, time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), f1=np.mean(f1), f2=np.mean(f2),pe=np.mean(p_e), re=np.mean(r_e), pl=np.mean(p_l), rl=np.mean(r_l))
            f.write(result)
        #model.save_weights(args.weight_path)
    else:
        print('==================== Training model ===================')
        print(args.pretrain_weight_path)
        if not args.pretrain_weight_path == 'None':
            model.load_weights(args.pretrain_weight_path, by_name=True)
        if args.data_config == 'ucm_si' or args.data_config == 'aid_si':
            # Prototype learning on single-scene aerial image datasets (e.g., UCM & AID)
            model.compile(optimizer=Nadam(lr=args.lr), loss='categorical_crossentropy', metrics=['acc'])
            model_checkpoint= ModelCheckpoint(args.weight_path, monitor="val_acc", save_best_only=True, save_weights_only=True, mode='max')
        else:
            # Competitor: CNN*
            model.compile(optimizer=Nadam(lr=args.lr), loss='binary_crossentropy', metrics=[F1_e, F2_e])
            model_checkpoint= ModelCheckpoint(args.weight_path, monitor="val_F1_e", save_best_only=True, save_weights_only=True, mode='max')

        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=2, min_lr=0.5e-15)
        model.fit(X_tr, y_tr, batch_size=args.bs, epochs=args.ep, validation_split=0.01, shuffle=True, callbacks=[model_checkpoint, lr_reducer])

if __name__ == '__main__':
    main()
