from backbone.model_irse import IR_50
from attack.privacy import FIM
import numpy as np
import argparse
import datetime
import torch
import sys
import cv2
import os


def main(args):

    # initialize the surrogate model
    model = IR_50([112, 112])
    model.load_state_dict(torch.load(args.pretrained, map_location=args.device))

    # make the output dir of privacy masks
    if not os.path.exists(args.adv_out):
        os.makedirs(args.adv_out)

    # load the list of images
    img_list = open(args.target_lst)
    files = img_list.readlines()

    np.random.seed(0)
    for protectee_id in range(args.num_protectees):
        # prepare training images
        imgs = np.ones((args.num_shot, 3, 112, 112), dtype='float32')  # each identity has 10 training images
        for j in range(args.num_shot):
            name = files[protectee_id * args.batch_size + j]  # use args.batch_size for generation
            img_name = os.path.join(args.data_dir, name)
            img_name = img_name.split('\n')[0]
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            imgs[j, :, :, :] = img
        imgs = torch.from_numpy(imgs).to(args.device)
        print('[Protectee %d] Training Statistics' % protectee_id)
        print('  > Training image tensor shape: %s (RGB)' % str(tuple(imgs.shape)))
        print('  > Training image tensor range: [%.2f, %.2f]' % (float(imgs.min()), float(imgs.max())))

        # start generating the protection mask
        start_time = datetime.datetime.now()
        fim = FIM(args.round, args.alpha, args.step_size, True, args.loss_type, args.nter, args.upper,
                  args.lower, args.device)
        noise = fim.process(model, imgs)  # use the attack function
        end_time = datetime.datetime.now()
        print("  > Time consumed: %s" % (end_time - start_time))

        # save the mask
        noise = noise[0].cpu().detach().numpy()
        save_path = os.path.join(args.adv_out, 'mask_id-%d.npy' % protectee_id)
        print('  > Mask shape=%s saved to %s' % (str(noise.shape), save_path))
        np.save(save_path, noise)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', help='surrogate model',
                        default='../../models/surrogate/IR_50-ArcFace-casia/Backbone_IR_50_Epoch_73_Batch_138000_Time_2020-05-07-23-48_checkpoint.pth')
    parser.add_argument('--adv_out', help='output dir of privacy masks', default='../../results')
    parser.add_argument('--target_lst', help='list of training images',
                        default='../../data/Privacy_common/privacy_train_v3_10.lst')
    parser.add_argument('--data_dir', help='dir of training images', default='../../data/Privacy_common')
    parser.add_argument('--batch_size', type=int, help='number of samples to generate a mask', default=10)
    parser.add_argument('--num_shot', type=int, help='each identity has 10 training images', default=10)
    parser.add_argument('--nter', type=int, help='initial iterations of convexhull', default=100)
    parser.add_argument('--upper', type=float, help='upper bound of reducedhull', default=1.0)
    parser.add_argument('--lower', type=float, help='lower bound of reducedhull', default=0.0)
    parser.add_argument('--loss_type', type=int, help='type of approximation method:0-->FI-UAP; '
                                                      '2-->FI-UAP+; 7-->OPOM-ClassCenter; 8-->OPOM-AffineHull;'
                                                      '9-->OPOM-ConvexHull', default=9)
    parser.add_argument('--alpha', type=float, help='perturbation budeget', default=8)
    parser.add_argument('--step_size', type=float, help='gradient step size, defalt 1 in this work', default=1)
    parser.add_argument('--round', type=int, help='training iterations', default=50)
    parser.add_argument('--device', type=str, help='device', default='cpu')
    parser.add_argument('--num_protectees', type=int, help='number of protectees', default=100)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
