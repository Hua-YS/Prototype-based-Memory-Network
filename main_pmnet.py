import os
import sys
import time
sys.path.append(os.path.abspath('./utils'))
import argparse
from mai import *
from pmnet import *


parser = argparse.ArgumentParser(description='si2mu')
parser.add_argument('--backbone', default='resnet50', metavar='MODEL', help='vggnet16, inceptionv3, resnet50, nasnet (default: resnet50)')
parser.add_argument('--weight_path', default='weights/aid2mai/pm-resnet50.h5')
parser.add_argument('--pretrain_weight_path', default='weights/aid2mai/resnet50-aid.h5')
parser.add_argument('--data_config', default='aid2mai', metavar='DATA', help='ucm2mai, aid2mai (default: aid2mai)')
parser.add_argument('--patch_size', default=224, type=int, metavar='N', help='image size (default: 224)')
parser.add_argument('--embed_dim', default=256, type=int, metavar='M', help='channel dimension of key, value, query (default: 256)')
parser.add_argument('--nb_head', default=20, type=int, metavar='N', help='number of heads in read controller (default: 20)')
parser.add_argument('--lr', default=2e-4, type=float, metavar='LR', help='learning rate (default: 2e-4)')
parser.add_argument('--bs', default=20, type=int, metavar='BS', help='batch size (default: 20)')
parser.add_argument('--ep', default=100, type=int, metavar='EP', help='training epochs (default: 100)')
parser.add_argument('--frozen', default=True, help='is backbone trainable (default: True)', action='store_false')
parser.add_argument('--evaluate', default=1, type=int, help='evaluate model (default: True)')


# ************************* Configuretion **************************
gpu_config(0, 0.3)
def main():
    global args
    args = parser.parse_args()

    print('==================== Loading data ====================')
    X_tr, y_tr, X_val, y_val = load_data(args.data_config, args.patch_size, args.evaluate)
    print('Data config:', args.data_config)
    print('Training data:', len(X_tr))
    print('Test data:', len(X_val))
    print(np.shape(y_val))
    print('==================== Building model ===================')
    model = pmnet(backbone=args.backbone, nb_classes=y_val.shape[-1], patch_size=args.patch_size, embed_channel=args.embed_dim, trainable=args.frozen, nb_head=args.nb_head)
    model.summary()
    #mem_file = 'memory/{}/memory_{}_tripletloss.mat'.format(args.data_config, args.backbone)
    mem_file = 'memory/{}/memory_{}.mat'.format(args.data_config, args.backbone)
    f = sio.loadmat(mem_file)
    memory = np.expand_dims(f['mean_feat'], 0)
    print('Backbone:', args.backbone)
    print('Memory: {}, Size: {}'.format(mem_file, np.shape(memory)))

    if args.evaluate:
        print('==================== Evaluating model ===================')
        model.load_weights(args.weight_path)#, by_name=True)
        out = model.predict([X_val, np.repeat(memory, len(X_val), axis=0)])
        p_e, r_e, p_l, r_l = PR_score(out, y_val)
        f1 = F_score(out, y_val)
        f2 = F_score(out, y_val, True, 2)
        print('f1 | f2 | pe | re | pl | rl \n {f1:.4f} & {f2:.4f} & {pe:.4f} & {re:.4f} & {pl:.4f} & {rl:.4f}'.format(f1=np.mean(f1), f2=np.mean(f2),pe=np.mean(p_e), re=np.mean(r_e), pl=np.mean(p_l), rl=np.mean(r_l)))
        with open('results.txt', 'a+') as f:
            result = 'weights: {weights:s} \ntime: {time:s} \nresults: f1 | f2 | pe | re | pl | rl \n{f1:.4f} & {f2:.4f} & {pe:.4f} & {re:.4f} & {pl:.4f} & {rl:.4f} \n'.format(weights=args.weight_path, time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), f1=np.mean(f1), f2=np.mean(f2),pe=np.mean(p_e), re=np.mean(r_e), pl=np.mean(p_l), rl=np.mean(r_l))
            f.write(result)

        sio.savemat(args.weight_path[:-3]+'.mat', {'gt':y_val,'pred':out})

    else:
        print('==================== Training model ===================')
        model.load_weights(args.pretrain_weight_path, by_name=True)
        model.compile(optimizer=Nadam(lr=args.lr), loss='binary_crossentropy', metrics=[F1_e, F12_e])
        model_checkpoint= ModelCheckpoint(args.weight_path, monitor="val_F12_e", save_best_only=True, save_weights_only=True, mode='max')
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=2, min_lr=0.5e-15)
        model.fit([X_tr, np.repeat(memory, len(X_tr), axis=0)], y_tr, batch_size=args.bs, epochs=args.ep, validation_split=0.1, shuffle=True, callbacks=[model_checkpoint, lr_reducer])
        model.save_weights(args.weight_path[:-3]+'_end.h5')

if __name__ == '__main__':
    main()
