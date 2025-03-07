import random
import time
import argparse
import shutil
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import numpy as np

import utils
from tllib.self_training.noise_adaptation import Noise_Adaptation
from tllib.vision.transforms import MultipleApply
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.data import ForeverDataIterator
from tllib.utils.logger import CompleteLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    log_path = args.log + args.data.lower() + "/seed_" + str(args.seed) + "/" + args.arch.lower()
    print("Log path: " + log_path)
    logger = CompleteLogger(log_path, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    cudnn.benchmark = False

    # Load data
    weak_augment = utils.get_train_transform(args.train_resizing, random_horizontal_flip=True,
                                             norm_mean=args.norm_mean, norm_std=args.norm_std)
    strong_augment = utils.get_train_transform(args.train_resizing, random_horizontal_flip=True,
                                               auto_augment=args.auto_augment,
                                               norm_mean=args.norm_mean, norm_std=args.norm_std)
    labeled_train_transform = MultipleApply([weak_augment, strong_augment])
    unlabeled_train_transform = MultipleApply([weak_augment, strong_augment])
    val_transform = utils.get_val_transform(args.val_resizing, norm_mean=args.norm_mean, norm_std=args.norm_std)
    print('labeled_train_transform: ', labeled_train_transform)
    print('unlabeled_train_transform: ', unlabeled_train_transform)
    print('val_transform:', val_transform)
    labeled_train_dataset, unlabeled_train_dataset, val_dataset = \
        utils.get_dataset(args.data,
                          args.num_samples_per_class,
                          args.root, labeled_train_transform,
                          val_transform,
                          unlabeled_train_transform=unlabeled_train_transform,
                          seed=args.seed)
    print("labeled_dataset_size: ", len(labeled_train_dataset))
    print('unlabeled_dataset_size: ', len(unlabeled_train_dataset))
    print("val_dataset_size: ", len(val_dataset))

    labeled_train_loader = DataLoader(labeled_train_dataset, batch_size=args.batch_size, shuffle=True, 
                                      num_workers=args.workers, drop_last=True)
    unlabeled_train_loader = DataLoader(unlabeled_train_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.workers, drop_last=True)
    labeled_train_iter = ForeverDataIterator(labeled_train_loader)
    unlabeled_train_iter = ForeverDataIterator(unlabeled_train_loader)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrained_checkpoint=args.pretrained_backbone)
    num_classes = labeled_train_dataset.num_classes
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = utils.ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim, pool_layer=pool_layer,
                                       finetune=args.finetune).to(device)
    
    # Generate noise
    print("Generate noise")
    mus = torch.randn([num_classes, args.bottleneck_dim])
    mus = nn.Parameter(mus)
    mus_labels = torch.arange(num_classes).reshape(-1)

    
    all_parameters = [{"params": mus}] + classifier.get_parameters()

    # Define optimizer and lr scheduler
    if args.lr_scheduler == 'exp':
        optimizer = SGD(all_parameters, args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
        lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    else:
        optimizer = SGD(all_parameters, args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
        lr_scheduler = utils.get_cosine_scheduler_with_warmup(optimizer, args.epochs * args.iters_per_epoch)


    if args.phase == 'test':
        checkpoint = torch.load(logger.get_checkpoint_path('20'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
        acc1, avg = utils.validate(val_loader, classifier, args, device, num_classes)
        print(acc1)
        return


    # Start training
    best_acc1 = 0.0
    best_avg = 0.0
    for epoch in range(args.epochs):
        # Print lr
        print(lr_scheduler.get_lr())

        # Train for one epoch
        train(labeled_train_iter, unlabeled_train_iter, classifier, optimizer, lr_scheduler, epoch, args, mus, mus_labels)

        # Evaluate on validation set
        acc1, avg = utils.validate(val_loader, classifier, args, device, num_classes)

        # Remember best acc@1 and save checkpoint
        if (epoch + 1) % 5 == 0:
            name = str(epoch + 1)
            torch.save(classifier.state_dict(), logger.get_checkpoint_path(name))

        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
        best_avg = max(avg, best_avg)

    print("best_acc1 = {:3.2f}".format(best_acc1))
    print('best_avg = {:3.2f}'.format(best_avg))
    logger.close()



def train(labeled_train_iter: ForeverDataIterator, unlabeled_train_iter: ForeverDataIterator, model, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace, mus, mus_labels):
    batch_time = AverageMeter('Time', ':2.2f')
    data_time = AverageMeter('Data', ':2.1f')
    cls_losses = AverageMeter('Cls Loss', ':3.2f')
    self_training_losses = AverageMeter('Noise Adaptation Loss', ':3.2f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    noise_accs = AverageMeter('Noise Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_losses, self_training_losses, cls_accs, noise_accs],
        prefix="Epoch: [{}]".format(epoch))

    # Switch to train mode
    model.train()

    end = time.time()
    batch_size = args.batch_size
    self_training_criterion = Noise_Adaptation()
    
    for i in range(args.iters_per_epoch):
        (x_l, x_l_strong), labels_l = next(labeled_train_iter)
        x_l = x_l.to(device)
        x_l_strong = x_l_strong.to(device)
        labels_l = labels_l.to(device)

        (x_u, x_u_strong), labels_u = next(unlabeled_train_iter)
        x_u = x_u.to(device)
        x_u_strong = x_u_strong.to(device)
        labels_u = labels_u.to(device)

        # Measure data loading time
        data_time.update(time.time() - end)

        # Clear grad
        optimizer.zero_grad()

        # Compute loss
        # Cross entropy loss
        y_l, y_l_feature = model(x_l, return_feature=True)
        y_l_strong, y_l_strong_feature = model(x_l_strong, return_feature=True)
        cls_loss = F.cross_entropy(y_l, labels_l) + args.trade_off_cls_strong * F.cross_entropy(y_l_strong, labels_l)

        # Prepare for unlabeled data
        y_u, y_u_feature = model(x_u, return_feature=True)
        y_u_strong, y_u_strong_feature = model(x_u_strong, return_feature=True)
        
        # Prepare for noise
        m_s = mus.to(device) # (C, D)
        m_s_labels = mus_labels.to(device)

        # Cross entropy loss
        m_s_predict = model.head(m_s)
        noise_cls_loss = F.cross_entropy(m_s_predict, m_s_labels)

        # NA loss
        self_training_loss = self_training_criterion(y_u_strong, y_u_strong_feature, y_u, y_u_feature, y_l_strong, y_l_strong_feature, y_l, y_l_feature, labels_l, m_s, device, args.PAM)

        # Measure accuracy and record loss
        loss = cls_loss + noise_cls_loss + args.trade_off_NA_training * self_training_loss

        loss.backward()

        losses.update(loss.item(), batch_size)
        cls_losses.update(cls_loss.item(), batch_size)
        self_training_losses.update(self_training_loss.item(), batch_size)

        noise_acc = accuracy(m_s_predict, m_s_labels)[0]
        noise_accs.update(noise_acc.item())
        cls_acc = accuracy(y_l, labels_l)[0]
        cls_accs.update(cls_acc.item(), batch_size)

        # Compute gradient and do SGD step
        optimizer.step()
        lr_scheduler.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Noise Adaptation for Semi-Supervised Learning')
    # Dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA',
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()))
    parser.add_argument('--num-samples-per-class', default=4, type=int,
                        help='number of labeled samples per class (default: 4)')
    parser.add_argument('--train-resizing', default='default', type=str)
    parser.add_argument('--val-resizing', default='default', type=str)
    parser.add_argument('--norm-mean', default=(0.485, 0.456, 0.406), type=float, nargs='+',
                        help='normalization mean')
    parser.add_argument('--norm-std', default=(0.229, 0.224, 0.225), type=float, nargs='+',
                        help='normalization std')
    parser.add_argument('--auto-augment', default='rand-m10-n2-mstd2', type=str,
                        help='AutoAugment policy (default: rand-m10-n2-mstd2)')
    # Model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=utils.get_model_names(),
                        help='backbone architecture: ' + ' | '.join(utils.get_model_names()) + ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=1024, type=int,
                        help='dimension of bottleneck (default: 1024)')
    parser.add_argument('--no-pool', action='store_true', default=False,
                        help='no pool layer after the feature extractor')
    parser.add_argument('--pretrained-backbone', default=None, type=str,
                        help="pretrained checkpoint of the backbone "
                             "(default: None, use the ImageNet supervised pretrained backbone)")
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='whether to use 10x smaller lr for backbone')
    # Training parameters
    parser.add_argument('--trade-off-cls-strong', default=0.1, type=float,
                        help='the trade-off hyper-parameter of cls loss on strong augmented labeled data')
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, metavar='LR', dest='lr',
                        help='initial learning rate')
    parser.add_argument('--lr-scheduler', default='exp', type=str, choices=['exp', 'cos'],
                        help='learning rate decay strategy')
    parser.add_argument('--lr-gamma', default=0.001, type=float,
                        help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float,
                        help='parameter for lr scheduler')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W',
                        help='weight decay (default:5e-4)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run (default: 20)')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='number of iterations per epoch (default: 500)')
    parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N',
                        help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training ')
    parser.add_argument("--log", default='ERM', type=str,
                        help="where to save logs, checkpoints and debugging images")
    parser.add_argument("--phase", default='train', type=str, choices=['train', 'test'],
                        help="when phase is 'test', only test the model")
    # NA parameters
    parser.add_argument('--trade-off-NA-training', default=1, type=float,
                        help='the trade-off hyper-parameter of NA loss')
    parser.add_argument('--PAM', action='store_true', default=False,
                        help='PAM loss or PCM loss')

    args = parser.parse_args()
    main(args)