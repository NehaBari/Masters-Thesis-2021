import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklvq import GMLVQ
from timeit import timeit
from sklearn.model_selection import KFold
from itertools import chain
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    cross_val_score,
    RepeatedKFold,
)
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve,roc_auc_score,auc
import colorama
from colorama import Fore, Style
matplotlib.rc("xtick", labelsize="small")
matplotlib.rc("ytick", labelsize="small")

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted, check_array


def getdata():
    data = pd.read_csv("feature_vectors.csv").to_numpy()
    labels = pd.read_csv("diagnosis_label.csv").to_numpy().squeeze()
    labelscenter = pd.read_csv("center_label.csv").to_numpy().squeeze()

    labelsfinal = labels + labelscenter

    # better way to perform this
    labelsdiseases1 = np.where(labelsfinal == 'HCUMCG')
    centerlabels1 = labelsfinal[labelsdiseases1]
    labelsdiseases2 = np.where(labelsfinal == 'HCUGOSM')
    # labelsdiseases = np.where(labels == 'HCCUN','1',centerlabels)
    centerlabels2 = labelsfinal[labelsdiseases2]
    labelsdiseases3 = np.where(labelsfinal == 'HCCUN')
    centerlabels3 = labelsfinal[labelsdiseases3]

    # better way to perform this
    centerlabels = np.concatenate((centerlabels1, centerlabels2, centerlabels3))
    centerdata = data[labelsdiseases1]
    centerdata = np.concatenate((centerdata, data[labelsdiseases2]))
    centerdata = np.concatenate((centerdata, data[labelsdiseases3]))

    scalar = StandardScaler()

    return data, labels, centerdata, centerlabels, labelscenter, scalar, labelsfinal


def preprocess_data(data, labels, labelscenter):
    # removing UMCG
    ######UMCG indices##########

    # ADindices = np.where(labels=='AD')

    UMCGindices = np.where(labelscenter == 'UMCG')

    # fetch umcg indices
    datanoUMCG = np.delete(data, UMCGindices, axis=0)

    # center
    # UMCG,CUN,UGOSM
    labelscenternoUMCG = np.delete(labelscenter, UMCGindices, axis=0)

    # get AD indices

    # disease
    # remove UMCG
    labelsdiseasenoUMCG = np.delete(labels, UMCGindices, axis=0)

    ### get AD indices
    ADindices = np.where(labelsdiseasenoUMCG == 'AD')

    ####################### Remove adindices from center and disease##################################
    centerdata_noAD = np.delete(datanoUMCG, ADindices, axis=0)

    # Final center labels after AD is removed
    centerlabels_noAD = np.delete(labelscenternoUMCG, ADindices, axis=0)

    # remove AD
    diseasedata_noAD = np.delete(datanoUMCG, ADindices, axis=0)
    # labelsnoAD for disease data
    diseaselabels_noAD = np.delete(labelsdiseasenoUMCG, ADindices, axis=0)

    ####################### Remove PD indices from center##################################

    PDindices = np.where(diseaselabels_noAD == 'PD')

    ################# FINAL Center data and labels###################

    final_center_data = np.delete(centerdata_noAD, PDindices, axis=0)

    final_center_labels = np.delete(centerlabels_noAD, PDindices, axis=0)

    ################# FINAL disease data and labels###################

    final_data_disease = diseasedata_noAD

    final_labels_disease = diseaselabels_noAD + centerlabels_noAD

    return final_center_data,final_center_labels, final_data_disease, final_labels_disease

def ztransform(data_ztransform,scalarz):
    # Sklearn's standardscaler to perform z-transform

    # Compute (fit) and apply (transform) z-transform
    # why do we do an inverse z transform
    #scalar = scalar
    data_ztransform = scalarz.fit_transform(data_ztransform)
    return data_ztransform,scalarz

class ProcessLogger:
    def __init__(self):
        self.states = np.array([])

    # A callback function has to accept two arguments, i.e., model and state, where model is the
    # current model, and state contains a number of the optimizers variables.
    def __call__(self, state):
        self.states = np.append(self.states, state)
        return False  # The callback function can also be used to stop training early,
        # if some condition is met by returning True.


def model_definition_center(logger):
    model = GMLVQ(
        distance_type="adaptive-squared-euclidean",
        activation_type="sigmoid",
        activation_params={"beta": 2},
        # wgd
        solver_type="wgd",
        solver_params={"max_runs": 50, "step_size": np.array([0.05, 0.03]), "callback": logger},
        # solver_params={"max_runs": 10,"batch_size":1,"step_size": np.array([0.1, 0.05])},
        random_state=1428, )

    return model





def model_definition_disease(correctionmatrix,logger):
    model1 = GMLVQ(
        distance_type="adaptive-squared-euclidean",
        activation_type="sigmoid",
        activation_params={"beta": 2},
        solver_type="wgd",
        solver_params={"max_runs": 50,"step_size": np.array([0.06, 0.03]),"callback": logger},
        #solver_params={"max_runs": 10,"batch_size":1,"step_size": np.array([0.1, 0.05])},
        random_state=1428,
        relevance_correction=correctionmatrix
    )
    return model1


def train_modelkfold(data, label, disease, correctionmatrix, repeated, scalar, folds):
    modelmatrix = np.zeros((repeated, folds), dtype=object)
    train_data = np.zeros((repeated, folds), dtype=object)
    train_labels = np.zeros((repeated, folds), dtype=object)
    # each entry will store the test labels and the predicted labels
    testlabels = np.zeros((repeated, folds), dtype=object)
    predicted = np.zeros((repeated, folds), dtype=object)
    probablities = np.zeros((repeated, folds), dtype=object)
    testing_indicesM = np.zeros((repeated, folds), dtype=object)
    traning_indicesM = np.zeros((repeated, folds), dtype=object)

    if disease == False:
        print('Repeated K fold for center data')
    else:
        print('Repeated K fold for disease data')
    for repeated in range(repeated):

        print("========Repeated fold number", str(repeated), "========")
        kfold = StratifiedKFold(folds, shuffle=True)

        for k, (training_indices, testing_indices) in enumerate(kfold.split(data, label)):
            logger = ProcessLogger()
            if disease == False:
                model = model_definition_center(logger)
                # print('in this')
                # print(type(model))
            else:
                # model = model_definition_center()
                model = model_definition_disease(correctionmatrix,logger)

            correct = 0
            trainX, trainY, testX, testY = data[training_indices], label[training_indices], data[testing_indices],label[testing_indices]

            # oversampling done and z transform applied
            oversample = RandomOverSampler(sampling_strategy='minority')
            trainX, trainY = oversample.fit_resample(trainX, trainY)

            #applying z transform on training data only
            pipeline = make_pipeline(scalar, model)
            pipeline.fit(trainX, trainY)
            predictedlabels = pipeline.predict(testX)

            ##############################train data############################

            modelmatrix[repeated, k] = pipeline[1]
            train_data[repeated, k] = pipeline.transform(trainX)
            train_labels[repeated, k] = trainY

            traning_indicesM[repeated, k] = training_indices
            ##############################test data############################
            probabilties = pipeline.predict_proba(testX)

            probablities[repeated, k] = probabilties

            testlabels[repeated, k] = testY

            predicted[repeated, k] = predictedlabels

            testing_indicesM[repeated, k] = testing_indices



    return modelmatrix, train_data, train_labels, testlabels, predicted, probablities, testing_indicesM, traning_indicesM

def calculate_lambda(modelmatrix, repeated):
    # average_lambda = np.mean(sum(model.lambda_ for model in modellist),axis=0)
    # check the shape of total_lambda

    total_lambda = np.zeros((36, 36), dtype='float')
    print(repeated)
    for index in range(repeated):
        print(index)
        lambdaperfold = np.zeros((36, 36), dtype='float')

        for modelin in modelmatrix[index, :]:
            lambdaperfold = lambdaperfold + modelin.lambda_

        lambdaperfold = lambdaperfold / np.shape(modelmatrix)[1]

        total_lambda = total_lambda + lambdaperfold

    total_lambda = total_lambda / repeated

    return total_lambda


def eigendecomposition(average_lambda):
    eigenvalues, eigenvectors = np.linalg.eigh(average_lambda)
    eigenvalues = np.flip(eigenvalues)
    eigenvectors = np.flip(eigenvectors, axis=1).T

    return eigenvalues, eigenvectors




def correction_matrix(eigvectorscenter):
    N=36
    K=1
    I = np.identity(N)
    outerproduct = np.zeros((N,N))
    for i in range(K):
        # check this
        outerproduct +=  np.outer(eigvectorscenter.T[:,i],eigvectorscenter[i,:])
        outerproduct1 =  np.outer(eigvectorscenter.T[:,0],eigvectorscenter[0,:])
        outerproduct2 =  np.outer(eigvectorscenter.T[:,1],eigvectorscenter[1,:])
    correctionmatrix = I-outerproduct
    return correctionmatrix



def calculate_prototype(modelmatrix,repeated,dimension,scalarmodel):
    #prototypeaverage = np.zeros((prototype,dimensions),dtype='float')
    numberofprototypes = len(modelmatrix[0][0].prototypes_)
    prototypeaverage = np.zeros((numberofprototypes,dimension),dtype='float')
    modellist = list(chain.from_iterable(zip(*modelmatrix)))
    scalarlist = list(chain.from_iterable(zip(*scalarmodel)))
    for modelin in range(len(modellist)):
        for i in range(numberofprototypes):
            prototypeaverage[i] = np.add(prototypeaverage[i],scalarlist[modelin].inverse_transform(modellist[modelin].prototypes_[i]))
    return prototypeaverage/len(modellist)



def diagonal_average_permodel(modelmatrix, dim):
    # element wise addition by default in numpy

    modellist = list(chain.from_iterable(zip(*modelmatrix)))
    for i in range(len(modellist)):
        if i == 0:
            data = np.diagonal(modellist[i].lambda_)
        else:
            data = np.vstack((np.diagonal(modellist[i].lambda_), data))
    avg = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    return avg, std


def check_orthogonality(centermodel,diseasemodel,dimensions,repeated,folds,eigenvectorscenter):
    centerlist = list(chain.from_iterable(zip(*centermodel)))
    disease_center = list()
    disease_averagecenter= list()
    for j in range(2):
        for array,modeld in enumerate(diseasemodel[j]):
            disease_averagecenter.append(np.dot(modeld.eigenvectors_[:2,:],eigenvectorscenter[:2,:].T))
            for modelc in centerlist:
                # eigenvectors are row vectors
                disease_center.append((np.dot(modeld.eigenvectors_[:2,:],modelc.eigenvectors_[:2,:].T)))
    return disease_averagecenter,disease_center


def transform1(X,eigenvaluesaverage,eigenvectoraverage,scale):
    X = check_array(X)
    eigenvaluesaverage[eigenvaluesaverage < 0] = 0
    eigenvectoraverage = np.sqrt(eigenvaluesaverage[:, None]) * eigenvectoraverage
    if scale:
        return np.matmul(X, eigenvectoraverage.T)
    return np.matmul(X, eigenvectoraverage.T)


def confusionmatrixc(test, predicted):
    CUNCUN = CUNUGOSM = UGOSMCUN = UGOSMUGOSM = 0
    CUNCUN, CUNUGOSM, UGOSMCUN, UGOSMUGOSM = confusion_matrix(test, predicted, labels=['CUN', 'UGOSM']).ravel()

    return CUNCUN, CUNUGOSM, UGOSMCUN, UGOSMUGOSM


def confusionmatrix_61classc(modelmatrix, testlabelmatrix, predictedlabelmatrix):

    np.seterr(divide='ignore', invalid='ignore')
    testlist = testlabelmatrix
    predictedlist = predictedlabelmatrix


    CUN_tpr = dict()
    CUN_fpr = dict()

    UGOSM_tpr = dict()
    UGOSM_fpr = dict()


    CUNCUN, CUNUGOSM, UGOSMCUN, UGOSMUGOSM = confusionmatrixc(testlabelmatrix, predictedlabelmatrix)

    nCUN_fpr = (UGOSMCUN) / (UGOSMCUN + UGOSMUGOSM)
    nCUN_tpr = (CUNCUN) / (CUNCUN + CUNUGOSM)

    nUGOSM_fpr = (CUNUGOSM) / (CUNUGOSM + CUNCUN)
    nUGOSM_tpr = (UGOSMUGOSM) / (UGOSMUGOSM + UGOSMCUN)

    #############################
    CUN_fpr = nCUN_fpr
    CUN_tpr = nCUN_tpr

    UGOSM_fpr = nUGOSM_fpr
    UGOSM_tpr = nUGOSM_tpr

    # return PDCUN_PDCUN,PDCUN_PDUGOSM,PDCUN_PDUMCG,PDCUN_ADUGOSM,PDCUN_ADUMCG,PDUGOSM_PDCUN, PDUGOSM_PDUGOSM, PDUGOSM_PDUMCG, PDUGOSM_ADUGOSM, PDUGOSM_ADUMCG,PDUMCG_PDCUN, PDUMCG_PDUGOSM, PDUMCG_PDUMCG, PDUMCG_ADUGOSM, PDUMCG_ADUMCG,ADUGOSM_PDCUN ,ADUGOSM_PDUGOSM ,ADUGOSM_PDUMCG, ADUGOSM_ADUGOSM ,ADUGOSM_ADUMCG,ADUMCG_PDCUN, ADUMCG_PDUGOSM, ADUMCG_PDUMCG, ADUMCG_ADUGOSM, ADUMCG_ADUMCG
    return CUN_fpr, CUN_tpr, UGOSM_fpr, UGOSM_tpr


def confusionmatrix_6classc(modelmatrix, testlabelmatrix, predictedlabelmatrix):
    testlist = list(chain.from_iterable(zip(*testlabelmatrix)))
    predictedlist = list(chain.from_iterable(zip(*predictedlabelmatrix)))
    # storing the values
    np.seterr(divide='ignore', invalid='ignore')

    # testlist= testlist.astype(int)
    # predictedlist = predictlist.astype(int)

    CUN_tpr = dict()
    CUN_fpr = dict()

    UGOSM_tpr = dict()
    UGOSM_fpr = dict()

    for i in range(len(testlist)):

        CUNCUN, CUNUGOSM, UGOSMCUN, UGOSMUGOSM = confusionmatrixc(testlist[i], predictedlist[i])

        nCUN_fpr = (UGOSMCUN) / (UGOSMCUN + UGOSMUGOSM)
        nCUN_tpr = (CUNCUN) / (CUNCUN + CUNUGOSM)

        nUGOSM_fpr = (CUNUGOSM) / (CUNUGOSM + CUNCUN)
        nUGOSM_tpr = (UGOSMUGOSM) / (UGOSMUGOSM + UGOSMCUN)

        #############################
        CUN_fpr[i] = nCUN_fpr
        CUN_tpr[i] = nCUN_tpr

        UGOSM_fpr[i] = nUGOSM_fpr
        UGOSM_tpr[i] = nUGOSM_tpr

    return CUN_fpr, CUN_tpr, UGOSM_fpr, UGOSM_tpr

# making the ceter dictionary


def center_dict(CUN_fpr,CUN_tpr,UGOSM_fpr,UGOSM_tpr,CUN_fpr1,CUN_tpr1,UGOSM_fpr1,UGOSM_tpr1,repeated,folds):

    n_classes=['CUN', 'UGOSM']
    plotdict_tpr1 = dict()
    plotdict_fpr1 = dict()

    plotdict_tpr1 = CUN_tpr1,UGOSM_tpr1
    plotdict_fpr1 = CUN_fpr1,UGOSM_fpr1


    n_classes=['CUN', 'UGOSM']
    plotdict_tpr = dict()
    plotdict_fpr = dict()


    for j in range(repeated*folds):


        plotdict_tpr[j]=CUN_tpr[j],UGOSM_tpr[j]
        plotdict_fpr[j]=CUN_fpr[j],UGOSM_fpr[j]
    return plotdict_tpr1,plotdict_fpr1,plotdict_tpr,plotdict_fpr

def plot_roc_center(testlabels,probabilties,testlabelsdsingle,probabiltysingle,plotdict_tpr1,plotdict_fpr1,plotdict_tpr,plotdict_fpr):

    testlist = list(chain.from_iterable(zip(*testlabels)))
    probabiltylist = list(chain.from_iterable(zip(*probabilties)))
    fpru = dict()
    tpru = dict()
    roc_aucu = dict()
    for k in range(len(testlist)):

        n_classes=['CUN', 'UGOSM']

        # structures
        fpr = dict()

        tpr = dict()

        roc_auc = dict()



        fpr1 = dict()

        tpr1 = dict()

        roc_auc1 = dict()


        colorrange = ['lightblue','lightblue']
        colorrange2 = ['red','green']
        for i in range(len(n_classes)):
            fpr[i], tpr[i], _ = roc_curve(testlist[k],probabiltylist[k][:,i],pos_label=n_classes[i], drop_intermediate=True)
            roc_auc[i] = auc(fpr[i], tpr[i])
            fpru[k] = fpr
            tpru[k] = tpr
            roc_aucu[k] = roc_auc

    for j in range(len(n_classes)):
            fpr1[j], tpr1[j], _ = roc_curve(testlabelsdsingle,probabiltysingle[:,j],pos_label=n_classes[j], drop_intermediate=True)
            roc_auc1[j] = auc(fpr1[j], tpr1[j])




    figsize=(10, 10)
    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)



    ax.plot([0, 1], [0, 1], 'k--')
    #ax.set_xlim([0.0, 1.0])
    #ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')


    for k1 in range(len(testlist)):
        for i1 in range(len(n_classes)):
            ax.plot(fpru[k1][i1], tpru[k1][i1], label='_nolegend_',color=colorrange[i1])

            #ax.plot(PDCUN_fpr[i], PDCUN_fpr[i],  marker="o", markersize="12")
            ax.plot(plotdict_fpr[k1][i1], plotdict_tpr[k1][i1],  color=colorrange[i1])#,marker="o", markersize="12")


    #plotdict_tpr1 = ADUGOSM_tpr1,ADUMCG_tpr1,PDCUN_tpr1,PDUGOSM_tpr1,PDUMCG_tpr1
    #plotdict_fpr1 = ADUGOSM_fpr1,ADUMCG_fpr1,PDCUN_tpr1,PDUGOSM_fpr1,PDUMCG_fpr1

    #for i2 in range(len(n_classes)):
    #    ax.plot(fpr1[i2], tpr1[1], label='ROC curve (AUC = %0.2f) for %s' % (roc_auc1[1], n_classes[1]),color=colorrange2[i2])
    #    ax.plot(plotdict_fpr1c[1], plotdict_tpr1[1], color=colorrange2[1],marker="o", markersize="12")

    ax.plot(fpr1[1], tpr1[1], label='ROC curve (AUC = %0.2f) for %s' % (roc_auc1[1], n_classes[1]),color=colorrange2[1])
    ax.plot(plotdict_fpr1[1], plotdict_tpr1[1], color=colorrange2[1],marker="o", markersize="12")

    ax.legend(loc="right")
    ax.grid(False)
    fig.savefig('roc6class.png')
    #ax.grid(alpha=.4)
    plt.show()



def confusionmatrixd(test, predicted):

    HCHC, HCDCUN, HCPDUGOSM, PDCUNHC, PDCUNPDCUN, PDCUNPDUGOSM, PDUGOSMHC, PDUGOSMPDCUN, PDUGOSMPDUGOSM = confusion_matrix(
        test, predicted, labels=['HC', 'PDCUN', 'PDUGOSM']).ravel()

    return HCHC, HCDCUN, HCPDUGOSM, PDCUNHC, PDCUNPDCUN, PDCUNPDUGOSM, PDUGOSMHC, PDUGOSMPDCUN, PDUGOSMPDUGOSM


def confusionmatrix_61class_d(testlabelmatrix, predictedlabelmatrix):

    np.seterr(divide='ignore', invalid='ignore')
    testlist = testlabelmatrix
    predictedlist = predictedlabelmatrix


    HC_tpr = dict()
    HC_fpr = dict()

    PDCUN_tpr = dict()
    PDCUN_fpr = dict()

    PDUGOSM_tpr = dict()
    PDUGOSM_fpr = dict()


    HCHC, HCPDCUN, HCPDUGOSM, PDCUNHC, PDCUNPDCUN, PDCUNPDUGOSM, PDUGOSMHC, PDUGOSMPDCUN, PDUGOSMPDUGOSM = confusionmatrixd(
        testlabelmatrix, predictedlabelmatrix)

    nHC_fpr = (PDCUNHC + PDUGOSMHC) / (PDCUNHC + PDUGOSMHC + PDCUNPDCUN + PDCUNPDUGOSM + PDUGOSMPDCUN + PDUGOSMPDUGOSM)
    nHC_tpr = (HCHC) / (HCHC + HCPDCUN + HCPDUGOSM)

    nPDCUN_fpr = (HCPDCUN + PDUGOSMPDCUN) / (HCPDCUN + PDUGOSMPDCUN + HCHC + HCPDUGOSM + PDUGOSMHC + PDUGOSMPDUGOSM)
    nPDCUN_tpr = (PDCUNPDCUN) / (PDCUNPDCUN + PDCUNHC + PDCUNPDUGOSM)

    nPDUGOSM_fpr = (HCPDUGOSM + PDCUNPDUGOSM) / (HCPDUGOSM + PDCUNPDUGOSM + HCHC + HCPDCUN + PDCUNPDCUN + PDCUNHC)
    nPDUGOSM_tpr = (PDUGOSMPDUGOSM) / (PDUGOSMPDUGOSM + PDUGOSMHC + PDUGOSMPDCUN)

    #############################
    HC_fpr = nHC_fpr
    HC_tpr = nHC_tpr

    PDCUN_fpr = nPDCUN_fpr
    PDCUN_tpr = nPDCUN_tpr

    PDUGOSM_fpr = nPDUGOSM_fpr
    PDUGOSM_tpr = nPDUGOSM_tpr


    return HC_fpr, HC_tpr, PDCUN_fpr, PDCUN_tpr, PDUGOSM_fpr, PDUGOSM_tpr


def confusionmatrix_6class_d(modelmatrix, testlabelmatrix, predictedlabelmatrix):
    testlist = list(chain.from_iterable(zip(*testlabelmatrix)))
    predictedlist = list(chain.from_iterable(zip(*predictedlabelmatrix)))
    # storing the values
    np.seterr(divide='ignore', invalid='ignore')



    HC_tpr = dict()
    HC_fpr = dict()

    PDCUN_tpr = dict()
    PDCUN_fpr = dict()

    PDUGOSM_tpr = dict()
    PDUGOSM_fpr = dict()

    for i in range(len(testlist)):

        HCHC, HCPDCUN, HCPDUGOSM, PDCUNHC, PDCUNPDCUN, PDCUNPDUGOSM, PDUGOSMHC, PDUGOSMPDCUN, PDUGOSMPDUGOSM = confusionmatrixd(
            testlist[i], predictedlist[i])

        nHC_fpr = (PDCUNHC + PDUGOSMHC) / (
                    PDCUNHC + PDUGOSMHC + PDCUNPDCUN + PDCUNPDUGOSM + PDUGOSMPDCUN + PDUGOSMPDUGOSM)
        nHC_tpr = (HCHC) / (HCHC + HCPDCUN + HCPDUGOSM)

        nPDCUN_fpr = (HCPDCUN + PDUGOSMPDCUN) / (HCPDCUN + PDUGOSMPDCUN + HCHC + HCPDUGOSM + PDUGOSMHC + PDUGOSMPDUGOSM)
        nPDCUN_tpr = (PDCUNPDCUN) / (PDCUNPDCUN + PDCUNHC + PDCUNPDUGOSM)

        nPDUGOSM_fpr = (HCPDUGOSM + PDCUNPDUGOSM) / (HCPDUGOSM + PDCUNPDUGOSM + HCHC + HCPDCUN + PDCUNPDCUN + PDCUNHC)
        nPDUGOSM_tpr = (PDUGOSMPDUGOSM) / (PDUGOSMPDUGOSM + PDUGOSMHC + PDUGOSMPDCUN)

        #############################
        HC_fpr[i] = nHC_fpr
        HC_tpr[i] = nHC_tpr

        PDCUN_fpr[i] = nPDCUN_fpr
        PDCUN_tpr[i] = nPDCUN_tpr

        PDUGOSM_fpr[i] = nPDUGOSM_fpr
        PDUGOSM_tpr[i] = nPDUGOSM_tpr

    return HC_fpr, HC_tpr, PDCUN_fpr, PDCUN_tpr, PDUGOSM_fpr, PDUGOSM_tpr


def disease_dict(HC_fpr, HC_tpr, PDCUN_fpr, PDCUN_tpr, PDUGOSM_fpr, PDUGOSM_tpr, HC_fpr1, HC_tpr1, PDCUN_fpr1,
                 PDCUN_tpr1, PDUGOSM_fpr1, PDUGOSM_tpr1, repeated, folds):
    n_classes = ['HC', 'PDCUN', 'PDUGOSM']
    plotdict_tpr1 = dict()
    plotdict_fpr1 = dict()

    plotdict_tpr1 = HC_tpr1, PDCUN_tpr1, PDUGOSM_tpr1
    plotdict_fpr1 = HC_fpr1, PDCUN_fpr1, PDUGOSM_fpr1



    n_classes = ['HC', 'PDCUN', 'PDUGOSM']
    plotdict_tpr = dict()
    plotdict_fpr = dict()


    for j in range(repeated * folds):

        plotdict_tpr[j] = HC_tpr[j], PDCUN_tpr[j], PDUGOSM_tpr[j]
        plotdict_fpr[j] = HC_fpr[j], PDCUN_fpr[j], PDUGOSM_fpr[j]

    return plotdict_tpr1,plotdict_fpr1,plotdict_tpr,plotdict_fpr








def plot_roc_disease(testlabels, probabilties,testlabelsdsingle,probabiltysingle,plotdict_tpr1,plotdict_fpr1,plotdict_tpr,plotdict_fpr):
    testlist = list(chain.from_iterable(zip(*testlabels)))
    probabiltylist = list(chain.from_iterable(zip(*probabilties)))
    fpru = dict()
    tpru = dict()
    roc_aucu = dict()
    for k in range(len(testlist)):

        n_classes = ['HC', 'PDCUN', 'PDUGOSM']

        # structures
        fpr = dict()

        tpr = dict()

        roc_auc = dict()

        fpr1 = dict()

        tpr1 = dict()

        roc_auc1 = dict()

        colorrange = ['green', 'red', 'blue']
        colorrange2 = ['green', 'red', 'blue']
        for i in range(len(n_classes)):
            # dictionary fpr tpr and roc calcualted per class per model and then stored in a bigger dict
            # Dict of Dict
            fpr[i], tpr[i], _ = roc_curve(testlist[k], probabiltylist[k][:, i], pos_label=n_classes[i],
                                          drop_intermediate=True)
            roc_auc[i] = auc(fpr[i], tpr[i])
            fpru[k] = fpr
            tpru[k] = tpr
            roc_aucu[k] = roc_auc

    for j in range(len(n_classes)):
        fpr1[j], tpr1[j], _ = roc_curve(testlabelsdsingle, probabiltysingle[:, j], pos_label=n_classes[j],
                                        drop_intermediate=True)
        roc_auc1[j] = auc(fpr1[j], tpr1[j])

    figsize = (10, 10)
    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot([0, 1], [0, 1], 'k--')
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')

    # for k1 in range(len(testlist)):
    #    for i1 in range(len(n_classes)):
    #        ax.plot(fpru[k1][i1], tpru[k1][i1], label='_nolegend_',color=colorrange[i1])

    # ax.plot(PDCUN_fpr[i], PDCUN_fpr[i],  marker="o", markersize="12")
    #        ax.plot(plotdict_fpr[k1][i1], plotdict_tpr[k1][i1],  color=colorrange[i1])#,marker="o", markersize="12")

    # plotdict_tpr1 = ADUGOSM_tpr1,ADUMCG_tpr1,PDCUN_tpr1,PDUGOSM_tpr1,PDUMCG_tpr1
    # plotdict_fpr1 = ADUGOSM_fpr1,ADUMCG_fpr1,PDCUN_tpr1,PDUGOSM_fpr1,PDUMCG_fpr1

    for i2 in range(len(n_classes)):
        ax.plot(fpr1[i2], tpr1[i2], label='ROC curve (AUC = %0.2f) for %s' % (roc_auc1[i2], n_classes[i2]),
                color=colorrange2[i2])
        ax.plot(plotdict_fpr1[i2], plotdict_tpr1[i2], color=colorrange2[i2], marker="o", markersize="12")

    ax.legend(loc="right")
    ax.grid(False)
    fig.savefig('roc6class.png')
    # ax.grid(alpha=.4)
    plt.show()
    return roc_aucu



def analytical_average_auc(roc_aucu):
    average_ruc = dict()
    # check this once
    for i in range(len(n_classes)):
        sum = 0
        for key in roc_aucu:
            sum += roc_aucu[key][i]
        average_ruc[i] = sum/len(roc_aucu)

    return average_ruc

# removing the z transform
def visualizeSinglemodeld(model, scalarI, data, labels, eigenvaluesaverage, eigenvectoraverage, prototypes1):
    transformed_data = model.transform(data, True)

    # all examples 1st and 2nd feature
    x_d = transformed_data[:, 0]
    y_d = transformed_data[:, 1]

    transformed_model = model.transform(prototypes1, scale=True)

    x_m = transformed_model[:, 0]
    y_m = transformed_model[:, 1]

    fig, ax = plt.subplots(figsize=(15, 15))
    fig.suptitle("Maindataset disease data with corresponding prototypes")

    colors = ['lightgreen', 'red', 'orange']
    colorsHC = ['lightblue', 'blue']
    # check the ordering from 4 class
    labelHC = ['HCUGOSM_early', 'HCCUN_late']
    labelss = ['HC', 'PDCUN_late', 'PDUGOSM_early']
    for i, cls in enumerate(model.classes_):
        ii = cls == labels
        if (i == 0):
            for hc in range(len(HCindices1)):
                # print('label0')
                ax.scatter(
                    x_d[HCindices1[hc]],
                    y_d[HCindices1[hc]],
                    c=colorsHC[hc],
                    s=100,
                    alpha=0.7,
                    edgecolors="white",
                    label=labelHC[hc])
                # label = model.classes_[model.prototypes_labels_[i]])
        else:
            ax.scatter(
                x_d[ii],
                y_d[ii],
                c=colors[i],
                s=100,
                alpha=0.7,
                edgecolors="white",
                label=labelss[i]
                # label=model.classes_[model.prototypes_labels_[i]]
            )

    # for i, txt in enumerate(HCindices):
    # ax.annotate(txt,(x_d[i],y_d[i]))
    # plt.scatter(x_d[ii],y_d[ii])

    for hc, txt in enumerate(HCindices):
        ax.annotate(txt, (x_d[HCindices[hc]], y_d[HCindices[hc]]))

    for hc1, txt1 in enumerate(labelss):
        ax.annotate(txt1, (x_m[hc1], y_m[HCindices[hc1]]), fontsize=20)

    ax.scatter(x_m, y_m, c=colors, s=500, alpha=0.8, edgecolors="black", linewidth=2.0)
    ax.set_xlabel("First eigenvector")
    ax.set_ylabel("Second eigenvector")
    ax.legend()
    # plt.ylim([-1, 30])
    # plt.xlim([-30, 30])
    ax.grid(True)
    print(model.classes_)
    # fig.savefig('visualize_center.eps', format='eps')
    fig.savefig('disease_correction_singelmodel.png')
    print('A1: Black, A2: Red, B1: Green, B2: Blue, Centers A and B, and diseases 1 and 2.) ')


# removing the z transform
def visualizeSinglemodelc(model, scalarI, data, labels, eigenvaluesaverage, eigenvectoraverage, prototypes1):
    # data = scalarI.inverse_transform(data)
    transformed_data = model.transform(data, True)

    # all examples 1st and 2nd feature
    x_d = transformed_data[:, 0]
    y_d = transformed_data[:, 1]

    # Transform the model, i.e., the prototypes (scaled by square root of eigenvalues "scale = True")
    # prototype inverser transform
    # prototypes = scalarI.inverse_transform(prototypes1)

    transformed_model = model.transform(prototypes1, scale=True)

    # transformed_model = transform1(prototypes,eigenvaluesaverage,eigenvectoraverage,scale=True)

    x_m = transformed_model[:, 0]
    y_m = transformed_model[:, 1]

    fig, ax = plt.subplots(figsize=(15, 15))
    fig.suptitle("Maindataset center data with corresponding prototypes")
    colors = ['pink', 'magenta']  # ,'red','lightgreen']
    for i, cls in enumerate(model.classes_):
        ii = cls == labels
        ax.scatter(
            x_d[ii],
            y_d[ii],
            c=colors[i],
            s=100,
            alpha=0.7,
            edgecolors="white",
            label=model.classes_[model.prototypes_labels_[i]],
        )
    # for i, txt in enumerate(HC_UGOSM_CUN):
    #    ax.annotate(txt,(x_d[i],y_d[i]))
    ax.scatter(x_m, y_m, c=colors, s=500, alpha=0.8, edgecolors="black", linewidth=2.0)
    ax.set_xlabel("First eigenvector")
    ax.set_ylabel("Second eigenvector")
    ax.legend()
    # plt.ylim([-1, 30])
    # plt.xlim([-30, 30])
    ax.grid(True)
    print(model.classes_)
    # fig.savefig('visualize_center.eps', format='eps')
    fig.savefig('center_correction_singelmodel.png')
    print('A1: Black, A2: Red, B1: Green, B2: Blue, Centers A and B, and diseases 1 and 2.) ')


# Plot the eigenvalues of the eigenvectors of the relevance matrix.
def ploteigenvalues(eigenvalues, eigenvectors, average_lambda, d):
    fig, ax = plt.subplots()
    fig.suptitle("Eigen values with correction " + d + " main data")
    ax.bar(range(0, len(eigenvalues)), eigenvalues)
    ax.set_ylabel("Weight")
    ax.grid(False)
    fig.savefig('eigenvalues' + d + '.eps', format='eps')
    fig.savefig('eigenvalues' + d + '.png')
    # fig.savefig('eigenvalues+d+.svg", format ='svg', dpi=1200)

    # Plot the first two eigenvectors of the relevance matrix, which  is called `omega_hat`.
    fig, ax = plt.subplots()
    fig.suptitle("First Eigenvector with correction " + d + " main data")
    ax.bar(feature_names, eigenvectors[0, :])
    ax.set_ylabel("Weight")
    ax.grid(False)
    fig.savefig('Firsteigenvector' + d + '.eps', format='eps')
    fig.savefig('Firsteigenvector' + d + '.png')

    fig, ax = plt.subplots()
    fig.suptitle("Second Eigenvector with correction " + d + " main data")
    ax.bar(feature_names, eigenvectors[1, :])
    ax.set_ylabel("Weight")
    ax.grid(False)
    fig.savefig('Secondeigenvector' + d + '.eps', format='eps')
    fig.savefig('Secondeigenvector' + d + '.png')
    fig, ax = plt.subplots()

    # The relevance matrix is available after fitting the model.
    relevance_matrix = average_lambda

    error = np.std(average_lambda_center, axis=0)
    fig.suptitle("Relevance Matrix Diagonal with correction " + d + " main data")
    ax.bar(feature_names, np.diagonal(relevance_matrix), yerr=error)
    ax.set_ylabel("Weight")
    ax.grid(False)
    fig.savefig('relevancematrix' + d + '.eps', format='eps')
    fig.savefig('relevancematrix' + d + '.png')


# Plot the eigenvalues of the eigenvectors of the relevance matrix.
def ploteigenvalueswithout(eigenvalues, eigenvectors, average_lambda, d):
    fig, ax = plt.subplots()
    fig.suptitle("Eigen values without correction " + d + " main data")
    ax.bar(range(0, len(eigenvalues)), eigenvalues)
    ax.set_ylabel("Weight")
    ax.grid(False)
    fig.savefig('eigenvalues' + d + '.eps', format='eps')
    fig.savefig('eigenvalues' + d + '.png')
    # fig.savefig('eigenvalues+d+.svg", format ='svg', dpi=1200)

    # Plot the first two eigenvectors of the relevance matrix, which  is called `omega_hat`.
    fig, ax = plt.subplots()
    fig.suptitle("First Eigenvector without correction " + d + " main data")
    ax.bar(feature_names, eigenvectors[0, :])
    ax.set_ylabel("Weight")
    ax.grid(False)
    fig.savefig('Firsteigenvector' + d + '.eps', format='eps')
    fig.savefig('Firsteigenvector' + d + '.png')

    fig, ax = plt.subplots()
    fig.suptitle("Second Eigenvector without correction " + d + " main data")
    ax.bar(feature_names, eigenvectors[1, :])
    ax.set_ylabel("Weight")
    ax.grid(False)
    fig.savefig('Secondeigenvector' + d + '.eps', format='eps')
    fig.savefig('Secondeigenvector' + d + '.png')
    fig, ax = plt.subplots()

    # plot diagonal, std and mean per feature for all dimensions
    relevance_matrix = average_lambda
    # means = np.mean(average_lambda_center,axis=0)
    # error = np.std(average_lambda_center,axis=0)
    error = average_lambad_disease_permodel
    fig.suptitle("Relevance Matrix Diagonal without correction " + d + " main data")
    ax.bar(feature_names, np.diagonal(relevance_matrix), yerr=error)
    ax.set_ylabel("Weight")
    ax.grid(False)
    fig.savefig('relevancematrix' + d + '.eps', format='eps')
    fig.savefig('relevancematrix' + d + '.png')
