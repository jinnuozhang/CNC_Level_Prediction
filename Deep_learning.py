import utils_t
from SpecDataset import form_dataset
# (swin_t, resnet18, wide_resnet50_2,vgg11,squeezenet1_0,densenet121,densenet161,densenet169,densenet201,resnet50,resnet101)
import torchvision.models as models
import torch.nn as nn
import torch
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from Common_Training import train_model
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix,roc_curve


if __name__ == '__main__':
    
    t_path = './train_hn'
    t_path1 = './train_ln'
    train_list,train_label = utils_t.get_file_path(t_path, t_path1)

    t_path = './test_hn'
    t_path1 = './test_ln'
    test_list,test_label = utils_t.get_file_path(t_path, t_path1)

    dataloader, dataset_size = form_dataset(train_list, 64, train_label, test_list, 64, test_label)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name_model2 = 'densenet201'
    model_n = models.densenet201

    model = utils_t.get_the_model(pre_defined = False, model = model_n, param_trainable = False, model_name = name_model2,
                    input_channel=9, class_num=2)

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    adam_optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    scheduler = lr_scheduler.StepLR(adam_optimizer, step_size=10, gamma=0.8)
    cls_criterion = nn.CrossEntropyLoss()
    num_epochs = 500

    #开始训练
    model_p1, Train_loss1, Valid_loss1, Test_loss1, Train_acc1,Valid_acc1, Test_acc1, best_acc1 = train_model(model=model, 
                                                                                                            criterion=cls_criterion, 
                                                                                                            optimizer=adam_optimizer, 
                                                                                                            num_epochs=num_epochs, 
                                                                                                            scheduler=scheduler, 
                                                                                                            dataloader=dataloader, device=device, 
                                                                                                            dataset_size=dataset_size,
                                                                                                            file_name=f'logs-freeze-{name_model2}', 
                                                                                                            save_mode=True)
    Train_accuracy = []
    for i in Train_acc1:
        Train_accuracy.append(i.detach().cpu().numpy())
    pd.DataFrame(Train_accuracy).to_excel(f'Train_acc1_{name_model2}.xlsx')

    Test_accuracy = []
    for i in Test_acc1:
        Test_accuracy.append(i.detach().cpu().numpy())
    pd.DataFrame(Test_accuracy).to_excel(f'Test_acc1_{name_model2}.xlsx')

    pd.DataFrame(Train_loss1).to_excel(f'Train_Loss1_{name_model2}.xlsx')
    pd.DataFrame(Test_loss1).to_excel(f'Test_Loss1_{name_model2}.xlsx')

    train_loss1 = pd.read_excel(f'Train_Loss1_{name_model2}.xlsx',index_col=0)
    test_loss1 = pd.read_excel(f'Test_Loss1_{name_model2}.xlsx',index_col=0)
    plt.plot(train_loss1,label='Train Loss')
    plt.plot(test_loss1,label='Test Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss-freeze-{name_model2}')
    plt.savefig(f'Loss-freeze-{name_model2}.png',dpi=600,bbox_inches='tight')

    train_acc1 = pd.read_excel(f'Train_acc1_{name_model2}.xlsx',index_col=0)
    test_acc1 = pd.read_excel(f'Test_acc1_{name_model2}.xlsx',index_col=0)
    plt.plot(train_acc1,label='Train Accuracy')
    plt.plot(test_acc1,label='Test Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy-freeze-{name_model2}')
    plt.savefig(f'Accuracy-freeze-{name_model2}.png',dpi=600,bbox_inches='tight')

    log_path = f'./logs-freeze-{name_model2}/valid-loss-0.5087-t_acc-68.90-v_acc-73.68-model.pth'
    model = utils_t.get_the_model(pre_defined = False, model = model_n, param_trainable = False, model_name = name_model2,
                    input_channel=9, class_num=2)

    r1 = utils_t.get_prediction_results(model, log_path, dataloader, device, model_name=f'{name_model2}_freeze')

