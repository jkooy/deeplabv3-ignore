import argparse
import os
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.metrics import Evaluator
from dataloaders.utils import decode_segmap



class Test(object):
    def __init__(self, args):
        self.args = args

        # configure datasetpath
        self.baseroot = None
        if args.dataset == 'pascal':
            self.baseroot = '/path/to/your/VOCdevkit/VOC2012/'
        ''' no support,
        # if you want train on these
        # you need modefy here 
        # refer to /dataloader/datasets/pascal to 
        #implement the corresponding constructor to dataset
        elif args.dataset == 'cityscapes':
            self.baseroot = '/path/to/your/cityscapes/'
        elif args.dataset == 'sbd':
            self.baseroot = '/path/to/your/sbd/'
        elif args.dataset == 'coco':
            self.baseroot = '/path/to/your/coco/'
        '''

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.test_loader, self.nclass = make_data_loader(self.baseroot, args, **kwargs)

        #define net model
        self.model = DeepLab(num_classes=self.nclass,
                          backbone=args.backbone,
                          output_stride=args.out_stride,
                          sync_bn=False,
                          freeze_bn=False).cuda()

        # self.model.module.load_state_dict(torch.load('./model_best.pth.tar', map_location='cpu'))
        self.evaluator = Evaluator(self.nclass)

        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        self.best_pred = 0.0

        if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        if args.cuda:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])

        self.best_pred = checkpoint['best_pred']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))



    def testing(self):
        tbar = tqdm(self.test_loader, desc='\r')
        self.model.eval()      #train_pattern
        # plt.ion()
        with torch.no_grad():
            for i, sample in enumerate(tbar):
                image, label = sample['image'], sample['label']
                if self.args.cuda:
                    image = image.cuda()
                output = self.model(image)     #(shape: (batch_size=1, n_class=21, img_h, img_w)

                #data visualization
                output = output.data.cpu().numpy()  #(shape: (batch_size=1, n_class=21, img_h, img_w)
                pred_label_imgs = np.argmax(output, axis=1) #(shape: (batch_size=1, img_h, img_w)
                pred_label_imgs = pred_label_imgs.astype(np.uint8)

                for index in range(pred_label_imgs.shape[0]):
                    pred_label_img = pred_label_imgs[index] #(shape: (img_h, img_w)
                    result = decode_segmap(pred_label_img, dataset='pascal', plot=False)

                    tmp = label[index].data.cpu().numpy()
                    tmp = tmp.astype(np.uint8)
                    segmap = decode_segmap(tmp, dataset='pascal')

                    img = image[index]  #(shape: (3, img_h, img_w)
                    img = img.data.cpu().numpy()
                    img = np.transpose(img, (1,2,0))    #(shape: (img_h, img_w, 3)
                    img *= (0.229, 0.224, 0.225)
                    img += (0.485, 0.456, 0.406)
                    img *= 255.0
                    img = img.astype(np.uint8)

                    # plt.figure()
                    plt.title('display')
                    plt.subplot(311)
                    plt.imshow(img)
                    plt.subplot(312)
                    plt.imshow(segmap)
                    plt.subplot(313)
                    plt.imshow(result)
                    plt.pause(0.7)
                    plt.clf()

            # plt.show()

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    # training hyper params
    parser.add_argument('--pattern', type=str, default='test',
                        choices=['train', 'test'],
                        help='train or test pattern)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')

    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0,1',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default='./run/pascal/deeplab-resnet/model_best.pth.tar',
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')


    if args.test_batch_size is None:
        args.test_batch_size = 1

    print(args)
    torch.manual_seed(args.seed)
    tester = Test(args)
    tester.testing()


if __name__ == "__main__":
   main()