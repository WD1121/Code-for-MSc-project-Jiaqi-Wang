import torchvision.transforms as transforms
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import argparse
import wandb
from models.ResNet import ResNet18

# Specify GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Training
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    train_acc = 0
    # Start iterating over the data in each batch
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # inputs:[b,3,32,32], targets:[b]
        # train_outputs:[b,10]
        inputs, targets = inputs.to(device), targets.to(device)
        # print(inputs.shape)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Calculate loss
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Calculate accuracy
        train_acc = correct / total
        # Print the loss and accuracy of the training set once every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print('[INFO] Epoch-{}-Batch-{}: Train: Loss-{:.4f}, Accuracy-{:.4f}'.format(epoch + 1,
                                                                                         batch_idx + 1,
                                                                                         loss.item(),
                                                                                         train_acc))
    # The accuracy of the training set in each epoch is calculated
    total_train_acc.append(train_acc)


# Testing
def test(epoch, ckpt):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_acc = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        test_acc = correct / total
        print(
            '[INFO] Epoch-{}-Test Accurancy: {:.3f}'.format(epoch + 1, test_acc), '\n')
        wandb.log({"test_acc": 100*test_acc})

    # The accuracy of test set
    total_test_acc.append(test_acc)

    # Save weight file
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, ckpt)
        best_acc = acc


if __name__ == '__main__':
    wandb.init(project="FL-epoch-50", name="Centralized DL standard resnet18")
    # Set hyperparameters
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/ResNet18-CIFAR10.pth')
    opt = parser.parse_args()

    # Set relevant parameters
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Set data enhancement
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root=opt.data, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=opt.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root=opt.data, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)



    # Load model
    print('==> Building model..')
    model = ResNet18().to(device)


    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # Load parameters of previous training
    if opt.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.checkpoint)
        model.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    #optimizer = optim.Adam(model.parameters())


    # Record acc of training and testing
    total_test_acc = []
    total_train_acc =[]

    # Start training
    for epoch in range(opt.epochs):
        train(epoch)
        test(epoch, opt.checkpoint)

    # Visualization
    plt.figure()
    plt.plot(range(opt.epochs), total_train_acc, label='Train Accuracy')
    plt.plot(range(opt.epochs), total_test_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('DL-ResNet18-CIFAR10-Accuracy')
    plt.legend()
    plt.savefig('output/ResNet18-CIFAR10-Accuracy.jpg')  # Automatically save the pictures
    plt.show()

    # Output best_acc
    print(f'Best Acc: {best_acc * 100}%')
    print('Training Done.')
