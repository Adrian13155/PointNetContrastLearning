from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize


# if __name__ == '__main__':
#     opt = parser.parse_args()
#     print(opt)
#     torch.manual_seed(random.randint(1, 10000))
#
#     dataset = ModelNetDataset(...)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
#
best_test_acc = 0.0
train_accuracies = []
test_accuracies = []

for epoch in range(opt.nepoch):
    start_epoch_time = time.time()  # 记录 epoch 开始时间

    for i, data in enumerate(dataloader, 0):
        batch_start_time = time.time()

        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()

        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()

        # 计算 batch ETA
        elapsed_epoch_time = time.time() - start_epoch_time
        avg_batch_time = elapsed_epoch_time / (i + 1)
        remaining_epoch_time = avg_batch_time * (len(dataloader) - i - 1)
        rem_mins, rem_secs = divmod(int(remaining_epoch_time), 60)

        if i % 10 == 0:
            print('[Epoch %d: %d/%d] train loss: %.4f accuracy: %.4f | ETA for epoch: %02d:%02d' %
                  (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize), rem_mins, rem_secs))

            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[Epoch %d: %d/%d] %s loss: %.4f accuracy: %.4f' %
                  (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

    # 调整位置：现在才更新 scheduler
    scheduler.step()
    print("Epoch %d finished in %.2f seconds | lr: %.6f" %
          (epoch, time.time() - start_epoch_time, optimizer.param_groups[0]['lr']))

    # 保存当前 epoch 模型（可选）
    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

    # 完整评估整个测试集
    total_correct = 0
    total_testset = 0
    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]

    test_acc = total_correct / float(total_testset)
    print("Epoch %d test accuracy: %.4f" % (epoch, test_acc))

    # 添加：记录本轮准确率
    train_acc = correct.item() / float(opt.batchSize)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)


    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch  # 新增：记录最优轮次
        torch.save(classifier.state_dict(), '%s/cls_model_best.pth' % opt.outf)
        print("New best model saved at epoch %d!" % epoch)

print("final best accuracy: %.4f" % best_test_acc)
print("Best test accuracy: %.4f at epoch %d" % (best_test_acc, best_epoch))

epochs = list(range(opt.nepoch))

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train vs Test Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_curve.png')
plt.show()
