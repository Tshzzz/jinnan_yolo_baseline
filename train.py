from src import build_yolov2
from src import yolo_box_encoder,group_encoder
from src import COCODataset
from src import build_yolov3


from torchvision import transforms
import config
import torch.utils.data as data


import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
from tensorboardX import SummaryWriter


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(100)


    transform = transforms.Compose([
        transforms.Resize(config.YOLO['image_size']),
        transforms.ToTensor(),
    ])
    anchor_wh = config.YOLO['anchor']
    featmap_size = config.YOLO['featmap_size']
    pretrained = config.YOLO['pretrain_model']
    epochs_start = config.epochs_start



    #net = build_yolov3(config.YOLO['class_num'], anchor_wh, featmap_size,do_detect = False,pretrained=pretrained)
    net = build_yolov2(config.YOLO['class_num'], anchor_wh, featmap_size, do_detect=False, pretrained=pretrained)
    if epochs_start > 0:
        net.load_state_dict(torch.load(config.save_dir + 'model_208.pkl'))
    net.cuda()
    net.train()





    encoder = yolo_box_encoder(anchor_wh, config.YOLO['class_num'], featmap_size)
    #encoder = group_encoder(anchor_wh, config.YOLO['class_num'], featmap_size)

    train_dataset = COCODataset('{}/jinnan_round1_train.json'.format(config.datasets_path),
                                    '{}/restricted/'.format(config.datasets_path),
                                    True, True, transform, encoder)

    train_loader = data.DataLoader(dataset=train_dataset,
                             batch_size=config.batch_size,
                             shuffle=True,
                             num_workers=8)

    optimizer = optim.SGD(net.parameters(), lr=config.start_lr, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[200], gamma=0.1)
    logger = SummaryWriter(config.save_dir)

    step = 0
    for epoch in range(epochs_start, 300):
        epoch_loss = 0
        train_iterator = tqdm(train_loader, ncols=30)
        mulit_batch_ = 0

        for train_batch, (images, target) in enumerate(train_iterator):

            images = images.cuda()
            loss_xx, loss_info = net(images,target)

            epoch_loss += loss_xx

            status = '[{0}] lr = {1} batch_loss = {2:.3f} epoch_loss = {3:.3f} '.format(
                epoch + 1, scheduler.get_lr()[0], loss_xx.data, epoch_loss.data / (train_batch + 1))

            train_iterator.set_description(status)


            if step % 10 == 0:
                for tag, value in loss_info.items():
                    logger.add_scalar(tag, value, step)
            loss_xx.backward()

            optimizer.step()
            optimizer.zero_grad()
            step += 1


        scheduler.step()

        if epoch > 100 and epoch % 5 == 0:
            torch.save(net.state_dict(), config.save_dir  + "model_{}.pkl".format(epoch))

        torch.save(net.state_dict(), config.save_dir + "model_.pkl")
