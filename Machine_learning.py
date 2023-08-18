import warnings
warnings.filterwarnings('ignore')
import os
%matplotlib inline
import matplotlib.pyplot as plt
import utils
import numpy as np
import pandas as pd
import seaborn as sns
import hyperspectral_pre_processing as hpp
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from scipy.signal import savgol_filter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix,roc_curve

def pre_data_transformation(data,reference_spectrum):
      data = data[:,:,20:320]
      mask = data[:, :, 0] != 0
      data = utils.auto_scale(data)
      data = utils.transmission_to_absorption(data)
      data = utils.msc_2d(data, reference_spectrum)
      data[~mask] = 0
      return data

if __name__ == '__main__':
    # read data
    bands = []
    for i in range(364):
        bands.append(utils.getWvFromPst(i, utils.paraPst2Wv))
    bands = np.array(bands)[20:320]
    data1 = pd.read_csv('./train_hn/average_spec.csv',header=None)
    data2 = pd.read_csv('./train_ln/average_spec.csv',header=None)
    data3 = pd.read_csv('./test_hn/average_spec.csv',header=None)
    data4 = pd.read_csv('./test_ln/average_spec.csv',header=None)
    # Preprocessing
    train_hn_data = np.array(data1)[:,20:320]
    train_ln_data = np.array(data2)[:,20:320]
    valid_hn_data = np.array(data3)[:,20:320]
    valid_ln_data = np.array(data4)[:,20:320]
    train_label = list(np.zeros(len(train_hn_data)).astype('int')) + list(np.ones(len(train_ln_data)).astype('int'))
    valid_label = list(np.zeros(len(valid_hn_data)).astype('int')) + list(np.ones(len(valid_ln_data)).astype('int'))
    transform = OneHotEncoder().fit_transform(np.array(train_label).reshape(-1,1))
    train_label_t = transform.toarray()
    train_x = np.concatenate((train_hn_data,train_ln_data),axis=0)
    valid_x = np.concatenate((valid_hn_data,valid_ln_data),axis=0)
    train_x = utils.auto_scale(train_x)
    valid_x = utils.auto_scale(valid_x)
    train_x = utils.transmission_to_absorption(train_x)
    valid_x = utils.transmission_to_absorption(valid_x)
    train_mean = np.mean(train_x,axis=0)
    train_x = utils.msc(train_x, train_mean)
    valid_x = utils.msc(valid_x, train_mean)
    train_x = savgol_filter(train_x, window_length=15,polyorder=1)
    valid_x = savgol_filter(valid_x, window_length=15,polyorder=1)
    # SVM
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'gamma':  [0.001, 0.01, 0.1, 10, 100]
    }
    svc = SVC(kernel='linear', probability=True)
    # svc = SVC(kernel='rbf', probability=True)
    grid_search = GridSearchCV(svc, param_grid=param_grid, cv=5)
    grid_search.fit(train_x, train_label)
    train_predict = grid_search.best_estimator_.predict(train_x)
    valid_predict = grid_search.best_estimator_.predict(valid_x)
    # Results
    print('Training results: {:.2f} %'.format(accuracy_score(train_predict,np.array(train_label))*100))
    print('Valid results: {:.2f} %'.format(accuracy_score(valid_predict,np.array(valid_label))*100))
    best_estimator = grid_search.best_estimator_
    print(best_estimator.get_params())
    precision = precision_score(np.array(valid_label), valid_predict)
    print("Precision:", precision)
    recall = recall_score(np.array(valid_label), valid_predict)
    print("Recall:", recall)
    f1 = f1_score(np.array(valid_label), valid_predict)
    print("F1 Score:", f1)
    y_scores = grid_search.best_estimator_.predict_proba(valid_x)[:, 1]
    fpr, tpr, thresholds = roc_curve(np.array(valid_label), y_scores)
    auc = roc_auc_score(np.array(valid_label), y_scores)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate',size=14)
    plt.ylabel('True Positive Rate',size=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve',size=14)
    plt.legend(fontsize=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.savefig('SVM ROC curve', bbox_inches='tight')
    plt.show()
    # Importance score
    best_estimator = grid_search.best_estimator_
    print(best_estimator.get_params())
    coef = best_estimator.coef_[0]
    plt.plot(abs(coef),alpha=0.5)
    peaks1, _ = find_peaks(abs(coef),distance=5,height=2,prominence=0.1)
    plt.scatter(peaks1,abs(coef[peaks1]),c='r',marker='*',s=50)
    plt.xticks(np.arange(0,300,20),bands[::20])
    plt.scatter(0,abs(coef[0]),c='r',marker='*',s=50)
    plt.xlabel('Wavelength(nm)')
    plt.ylabel('SVM Importance')
    plt.legend(['Importance score','Selected bands'])
    plt.savefig('Importance score.png',dpi=600,bbox_inches='tight')
    svm_bands = np.concatenate((np.array([bands[0]]),bands[peaks1]))
    pd.DataFrame(svm_bands).to_excel('./svm_bands_total.xlsx',index=False,header=False)