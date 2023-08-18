import time
import torch
import os
import copy
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

def entropy_minimization_loss(logits):
    """
    Computes entropy minimization loss.
    :param logits: raw, unscaled predictions from a neural network.
    :return: scalar entropy loss
    """
    probabilities = F.softmax(logits, dim=1)
    log_probabilities = F.log_softmax(logits, dim=1)
    entropy_loss = -(probabilities * log_probabilities).sum(dim=1).mean()
    return entropy_loss

def train_model(model, criterion, optimizer, num_epochs, scheduler, dataloader, device, dataset_size,file_name, save_mode=True):
    since = time.time()
    best_acc = 0.0
    tbest_acc = 0.0
    tbest_loss = 100
    tbest_epoch = 0
    Train_loss = []
    Valid_loss = []
    Test_loss = []
    Train_acc = []
    Valid_acc = []
    Test_acc = []
    Flag = True
    if file_name not in os.listdir('./'):
        os.makedirs(f'./{file_name}')
    for epoch in range(num_epochs):
        print('Epoch {}/{}-Trainable:{}'.format(epoch+1, num_epochs,Flag))
        print('-' * 50)
        for phase in ['train','valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                model.eval()
            running_loss = 0.0
            running_corrects = 0.0
            count = 0
            for inputs, label1 in dataloader[phase]:
                inputs = inputs.to(device).float()
                label = label1.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(inputs)
                    loss = criterion(output, label)
                    preds = torch.argmax(output,1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == label.data)
                count += len(inputs)
                if count == dataset_size[phase]:
                    print('>>>>>Stage-{} Progress-{}/{}>>>>>'.format(phase, count,dataset_size[phase]))
                
            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

            if phase == 'train':
                scheduler.step()
                Train_loss.append(epoch_loss)
                Train_acc.append(100 * epoch_acc)

            if phase == 'valid':
                Test_loss.append(epoch_loss)
                Test_acc.append(100 * epoch_acc)
                # if epoch_loss < tbest_loss or epoch_acc > tbest_acc:
                if epoch_acc > tbest_acc:
                    tbest_acc = epoch_acc
                    tbest_loss = epoch_loss
                    tbest_epoch = epoch
                    if save_mode:
                        torch.save(model.module.state_dict(), './{}/{}-loss-{:.4f}-t_acc-{:.2f}-v_acc-{:.2f}-model.pth'.format(file_name, phase, tbest_loss, Train_acc[-1], 100 * tbest_acc))
                        best_model_wts = copy.deepcopy(model.state_dict())
            print('Phase: {} --- Loss: {:.4f} --- Acc: {:.4f}%'.format(phase, epoch_loss, 100 * epoch_acc))
        print('-' * 50)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc and loss : {:2f} and {:2f} at epoch: {}'.format(100*tbest_acc,tbest_loss, tbest_epoch))
    model.load_state_dict(best_model_wts)
    return [model, Train_loss, Valid_loss, Test_loss, Train_acc, Valid_acc, Test_acc, best_acc]

def train_model_withaux(model, gama, criterion, optimizer, num_epochs, scheduler, dataloader, device, dataset_size,file_name, save_mode=True):
    since = time.time()
    best_acc = 0.0
    tbest_acc = 0.0
    tbest_loss = 100
    tbest_epoch = 0
    Train_loss = []
    Valid_loss = []
    Test_loss = []
    Train_acc = []
    Valid_acc = []
    Test_acc = []
    if file_name not in os.listdir('./'):
        os.makedirs(f'./{file_name}')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 50)
        for phase in ['train','valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                model.eval()
            running_loss = 0.0
            running_corrects = 0.0
            count = 0
            optimizer.zero_grad()
            for inputs, label in dataloader[phase]:
                inputs = inputs.to(device).float()
                label = label.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(inputs)
                    loss = criterion(output, label) + gama * entropy_minimization_loss(output)
                    preds = torch.argmax(output,1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == label.data)
                count += len(inputs)
                if count == dataset_size[phase]:
                    print('>>>>>Stage-{} Progress-{}/{}>>>>>'.format(phase, count,dataset_size[phase]))
        
            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

            if phase == 'train':
                scheduler.step()
                Train_loss.append(epoch_loss)
                Train_acc.append(100 * epoch_acc)
                
            if phase == 'valid':
                Test_loss.append(epoch_loss)
                Test_acc.append(100 * epoch_acc)
                # if epoch_loss < tbest_loss or epoch_acc > tbest_acc:
                if epoch_acc > tbest_acc:
                    tbest_acc = epoch_acc
                    tbest_loss = epoch_loss
                    tbest_epoch = epoch
                    if save_mode:
                        torch.save(model.module.state_dict(), './{}/{}-loss-{:.4f}-t_acc-{:.2f}-v_acc-{:.2f}-model.pth'.format(file_name, phase, tbest_loss, Train_acc[-1], 100 * tbest_acc))
                        best_model_wts = copy.deepcopy(model.state_dict())
            print('Phase: {} --- Loss: {:.4f} --- Acc: {:.4f}%'.format(phase, epoch_loss, 100 * epoch_acc))
            # print('Phase: {} --- Loss: {:.4f} L1:{:.4f} --- Acc: {:.4f}%'.format(phase, epoch_loss, epoch_loss1, 100 * epoch_acc))
        print('-' * 50)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc and loss : {:2f} and {:2f} at epoch: {}'.format(100*tbest_acc,tbest_loss, tbest_epoch))
    model.load_state_dict(best_model_wts)
    return [model, Train_loss, Valid_loss, Test_loss, Train_acc, Valid_acc, Test_acc, best_acc]