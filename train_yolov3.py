from src import build_yolov3
from src import group_encoder
from src import COCODataset
from torchvision import transforms
import torch.utils.data as data


import torch
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from tensorboardX import SummaryWriter
from yolov3_config import YOLO
import torch.nn as nn

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(100)


    transform = transforms.Compose([
        transforms.Resize(YOLO['image_size']),
        transforms.ToTensor(),
    ])
    anchor_wh = YOLO['anchor']
    featmap_size = YOLO['featmap_size']
    pretrained = YOLO['pretrain_model']
    epochs = YOLO['epochs']
    epochs_start = YOLO['epochs_start']
    lr = YOLO['start_lr']
    datasets_path = YOLO['datasets_path']
    bs = YOLO['batch_size']
    savedir = YOLO['save_dir']
    cls_num = YOLO['class_num']
    anno_path = YOLO['anno_path']
    steps = YOLO['steps']

    net = build_yolov3(cls_num, anchor_wh, featmap_size, do_detect=False, pretrained=pretrained)

    if epochs_start > 0:
        net.load_state_dict(torch.load(savedir + 'model_.pkl'))
    net.cuda()
    net.train()

    #encoder = yolo_box_encoder(anchor_wh, cls_num, featmap_size)
    encoder = group_encoder(anchor_wh, cls_num, featmap_size)

    train_dataset = COCODataset(anno_path,
                                    '{}/restricted/'.format(datasets_path),
                                    True, True, transform, encoder)

    train_loader = data.DataLoader(dataset=train_dataset,
                             batch_size=bs,
                             shuffle=True,
                             num_workers=8)

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    #scheduler = MultiStepLR(optimizer, milestones=steps, gamma=0.1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs // 9) + 1)

    logger = SummaryWriter(savedir)
    step = 0
    for epoch in range(epochs_start, epochs):
        epoch_loss = 0
        train_iterator = tqdm(train_loader, ncols=30)
        mulit_batch_ = 0

        for train_batch, (images, target) in enumerate(train_iterator):

            images = images.cuda()
            loss_xx, loss_info = net(images,target)

            epoch_loss += loss_xx
            status = '[{0}] lr = {1:.5f} batch_loss = {2:.3f} epoch_loss = {3:.3f} '.format(
                epoch + 1, scheduler.get_lr()[0], loss_xx.data, epoch_loss.data / (train_batch + 1))
            train_iterator.set_description(status)

            if step % 10 == 0:
                for tag, value in loss_info.items():
                    logger.add_scalar(tag, value, step)
            loss_xx.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.25)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

        scheduler.step()
        if epoch > 100 and epoch % 5 == 0:
            torch.save(net.state_dict(), savedir  + "model_{}.pkl".format(epoch))

        torch.save(net.state_dict(), savedir + "model_.pkl")
