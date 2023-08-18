import os
from torchvision.models import swin_t
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix,roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from tqdm import tqdm

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def get_file_path(t_path, t_path1,sort_mode = False):
    train_list = []
    train_list1 = []
    train_list2 = []
    train_label = []
    for i in os.listdir(t_path):
        if i.endswith('.npy'):
            train_list1.append(os.path.join(t_path, i))
            # train_label.append(0)
    train_label1 = [0] * len(train_list1)
    if sort_mode:
        train_list1.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    for j in os.listdir(t_path1):
        if j.endswith('.npy'):
            train_list2.append(os.path.join(t_path1, j))
            # train_label.append(1)
    train_label2 = [1] * len(train_list2)
    if sort_mode:
        train_list2.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    train_list.extend(train_list1)
    train_list.extend(train_list2)
    train_label.extend(train_label1)
    train_label.extend(train_label2)
    return train_list, train_label

def get_the_model(pre_defined = False, model = swin_t, 
                  param_trainable = False, model_name = 'swin_t',
                  input_channel=9, class_num=2):
    if pre_defined:
        for param in model.parameters():
            param.requires_grad = param_trainable
        return model
    else:
        model = model(weights='IMAGENET1K_V1')
        for param in model.parameters():
            param.requires_grad = param_trainable

        if model_name in ['swin_t', 'swin_s']:
            print('{} is used, trainable: {}'.format(model_name, param_trainable))
            model.features[0][0] =  nn.Conv2d(input_channel, 96, kernel_size=(4, 4), stride=(4, 4))
            model.head = nn.Linear(in_features=768, out_features=class_num, bias=True)
        elif model_name == 'swin_b':
            print('{} is used, trainable: {}'.format(model_name, param_trainable))
            model.features[0][0] =  nn.Conv2d(input_channel, 128, kernel_size=(4, 4), stride=(4, 4))
            model.head = nn.Linear(in_features=1024, out_features=class_num, bias=True)
        elif model_name == 'resnet18':
            print('{} is used, trainable: {}'.format(model_name, param_trainable))
            model.conv1 =  nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.fc = nn.Linear(in_features=512, out_features=class_num, bias=True)
        elif model_name in ['resnet50','resnet101', 'resnet152', 'wide_resnet50', 'wide_resnet101']:
            print('{} is used, trainable: {}'.format(model_name, param_trainable))
            model.conv1 =  nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.fc = nn.Linear(in_features=2048, out_features=class_num, bias=True)
        elif model_name == 'vgg11':
            print('{} is used, trainable: {}'.format(model_name, param_trainable))
            model.features[0] = nn.Conv2d(input_channel, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            model.classifier[6] = nn.Linear(in_features=4096, out_features=class_num, bias=True)
        elif model_name == 'squeezenet1_0':
            print('{} is used, trainable: {}'.format(model_name, param_trainable))
            model.features[0] = nn.Conv2d(input_channel, 96, kernel_size=(7, 7), stride=(2, 2))
            model.classifier[1] = nn.Conv2d(512, class_num, kernel_size=(1, 1), stride=(1, 1))
        elif model_name == 'densenet121':
            print('{} is used, trainable: {}'.format(model_name, param_trainable))
            model.features[0] = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.classifier = nn.Linear(in_features=1024, out_features=class_num, bias=True)
        elif model_name == 'densenet161':
            print('{} is used, trainable: {}'.format(model_name, param_trainable))
            model.features[0] = nn.Conv2d(input_channel, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.classifier = nn.Linear(in_features=2208, out_features=class_num, bias=True)
        elif model_name == 'densenet169':
            print('{} is used, trainable: {}'.format(model_name, param_trainable))
            model.features[0] = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.classifier = nn.Linear(in_features=1664, out_features=class_num, bias=True)
        elif model_name == 'densenet201':
            print('{} is used, trainable: {}'.format(model_name, param_trainable))
            model.features[0] = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.classifier = nn.Linear(in_features=1920, out_features=class_num, bias=True)
        return model
    
def softmax_row_wise(X):
    softmax_values = []
    for row in X:
        exponential_values = np.exp(row)
        sum_exponentials = np.sum(exponential_values)
        softmax_row = exponential_values / sum_exponentials
        softmax_values.append(softmax_row)
    return np.array(softmax_values)

def get_prediction_results_aux(model, log_path, dataloader, device, model_name):
    model = model.to(device)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model.load_state_dict(torch.load(log_path))
    model.eval()
    pred_result = []    # 预测结果
    true_result = []    # 真实结果
    score_results = []  # 预测分数
    for inputs, label1 in dataloader['valid']:
        inputs = inputs.to(device).float()
        label1 = label1.to(device)
        output1,output2 = model(inputs)
        score_results.append(output1.cpu().detach().numpy())
        preds = torch.argmax(output1,1)
        pred_result.append(preds.cpu().numpy())
        true_result.append(label1.cpu().numpy())
    scores = softmax_row_wise(score_results[0])[:,1]
    true_result = np.array(true_result[0])
    pred_result = np.array(pred_result[0])
    print('Start getting prediction metrics...')
    # Accuracy
    accuracy = accuracy_score(np.array(true_result), pred_result) * 100
    print('1. Testing Accuracy : {:.2f}%'.format(accuracy))
    # Precision
    precision = precision_score(np.array(true_result), pred_result)
    print('2. Precision: {:.2f}'.format(precision))
    # Recall
    recall = recall_score(np.array(true_result), pred_result)
    print('3. Recall: {:.2f}'.format(recall))
    # F1 Score
    f1 = f1_score(np.array(true_result), pred_result)
    print('4. F1 Score: {:.2f}'.format(f1))
    print('Start drawing confusion matrix...')
    cm = confusion_matrix(np.array(true_result), pred_result)
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(cm, annot=True, fmt='d',xticklabels=['High Nitrogen', 'Low Nitrogen'],yticklabels=['High Nitrogen', 'Low Nitrogen']
                    ,annot_kws={"fontsize":12})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    plt.xlabel('Predicted',size=14)
    plt.ylabel('Truth',size=14)
    plt.title('Confusion Matrix',size=14)
    plt.savefig(f'{model_name}_results.png', bbox_inches='tight')
    plt.show()
    
    print('Start drawing ROC curve...')
    # ROC Curve
    y_scores = scores
    fpr, tpr, thresholds = roc_curve(np.array(true_result), y_scores)
    auc = roc_auc_score(np.array(true_result), y_scores)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate',size=14)
    plt.ylabel('True Positive Rate',size=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve',size=14)
    plt.legend(fontsize=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.savefig(f'{model_name} ROC curve', bbox_inches='tight')
    plt.show()
    return [accuracy, precision, recall, f1, fpr,  tpr, thresholds, auc]

def get_prediction_results(model, log_path, dataloader, device, model_name, mode='valid'):
    model = model.to(device)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model.load_state_dict(torch.load(log_path))
    model.eval()
    pred_result = []    # 预测结果
    true_result = []    # 真实结果
    score_results = []  # 预测分数
    for inputs, label1 in dataloader[mode]:
        inputs = inputs.to(device).float()
        label1 = label1.to(device)
        output1 = model(inputs)
        score_results.append(output1.cpu().detach().numpy())
        preds = torch.argmax(output1,1)
        pred_result.append(preds.cpu().numpy())
        true_result.append(label1.cpu().numpy())

    score_results = flatten_list(score_results)
    true_result = flatten_list(true_result)
    pred_result = flatten_list(pred_result)

    scores = softmax_row_wise(score_results)[:,1]
    true_result = np.array(true_result)
    pred_result = np.array(pred_result)
    print('Start getting prediction metrics...')
    # Accuracy
    accuracy = accuracy_score(np.array(true_result), pred_result) * 100
    print('1. Testing Accuracy : {:.2f}%'.format(accuracy))
    # Precision
    precision = precision_score(np.array(true_result), pred_result)
    print('2. Precision: {:.2f}'.format(precision))
    # Recall
    recall = recall_score(np.array(true_result), pred_result)
    print('3. Recall: {:.2f}'.format(recall))
    # F1 Score
    f1 = f1_score(np.array(true_result), pred_result)
    print('4. F1 Score: {:.2f}'.format(f1))
    print('Start drawing confusion matrix...')
    cm = confusion_matrix(np.array(true_result), pred_result)
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(cm, annot=True, fmt='d',xticklabels=['High Nitrogen', 'Low Nitrogen'],yticklabels=['High Nitrogen', 'Low Nitrogen']
                    ,annot_kws={"fontsize":12})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    plt.xlabel('Predicted',size=14)
    plt.ylabel('Truth',size=14)
    plt.title('Confusion Matrix',size=14)
    plt.savefig(f'{model_name}_results.png', bbox_inches='tight')
    plt.show()
    
    print('Start drawing ROC curve...')
    # ROC Curve
    y_scores = scores
    fpr, tpr, thresholds = roc_curve(np.array(true_result), y_scores)
    auc = roc_auc_score(np.array(true_result), y_scores)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate',size=14)
    plt.ylabel('True Positive Rate',size=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve',size=14)
    plt.legend(fontsize=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.savefig(f'{model_name} ROC curve', bbox_inches='tight')
    plt.show()
    return [pred_result, accuracy, precision, recall, f1, fpr,  tpr, thresholds, auc]

def transform_data(ln_path, folder_name, calibration=False, mode='train', reference='white', green_path=None, cal_heatmap=False, resize=False, resize_shape=(128,96)):
    ndvi_theshold = 0.4
    count = 0
    green_count = 0
    if folder_name not in os.listdir():
        os.makedirs('./{}'.format(folder_name))
    aver_spec = []
    for i in tqdm(ln_path):
        data = np.load(i)
        loc_path = '.' + i.split('.')[0] + i.split('.')[1] + '_imgPst.' + i.split('.')[2]
        if reference == 'white':
            ref_path = '.' + i.split('.')[0] + i.split('.')[1] + '_whRef.' + i.split('.')[2]
            w_ref = np.load(ref_path)
            ref = np.expand_dims(w_ref,axis=1)
        elif reference == 'green':
            g_ref = np.load(green_path[green_count])
            green_count += 1
            ref = np.expand_dims(g_ref,axis=1)
        motion_loc = np.load(loc_path)
        data = utils.get_interplation(motion_loc,data)
        data = data / ref
        if reference == 'green':
            calibration_matrix = np.array(pd.read_csv('./mean_grf_specs_03012023.csv',header=None))
            calibration_matrix = np.expand_dims(calibration_matrix,axis=0)
            data = data*calibration_matrix
        ndvi_heatmap = utils.get_ndvi(data, utils.paraWv2Pst)
        leaf_mask = ndvi_heatmap > ndvi_theshold
        data = utils.cali_rawdata(leaf_mask, data, mode=calibration)
        if cal_heatmap:
            mask1 = leaf_mask.astype('uint8')
            contours, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for j in range(len(contours)):
                if len(contours[j]) > 100:
                    cnt = contours[j]
            x,y,w,h = cv2.boundingRect(cnt)
            data = data[y:y+h,x:x+w]
            # new_data = new_data[:,10:,:]
            mask1 = data[:,:,0] != 0
            data = data[:,-880:,:]
        leaf_mask = data[:,:,0] != 0
        # plt.imshow(leaf_mask)
        # plt.show()
        if mode == 'average':
            aver_spec.append(np.mean(data[leaf_mask,:],axis=0))
            continue
        if mode == 'train':
            img_list = utils.get_five_partition_data(data)
            img6 = utils.get_average_data(data)
            img_list.append(img6)
            for img in img_list:
                if resize:
                    img = cv2.resize(img, resize_shape)
                np.save('./{}/{}.npy'.format(folder_name,count),img)
                count+=1
        if mode == 'test':
            img6 = utils.get_average_data(data)
            if resize:
                img6 = cv2.resize(img6, resize_shape)
            np.save('./{}/{}.npy'.format(folder_name,count),img6)
            count+=1
    if mode == 'average':
        pd.DataFrame(aver_spec).to_csv('./{}/average_spec.csv'.format(folder_name),index=False,header=False)