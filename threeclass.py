import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklvq import GMLVQ
from itertools import chain
from sklearn.metrics import confusion_matrix

from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, auc
import seaborn as sns

matplotlib.rc("xtick", labelsize="small")
matplotlib.rc("ytick", labelsize="small")

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_array



params = {'legend.fontsize': 'medium',
          'figure.figsize': (23, 10),
          'axes.labelsize': 'large',
          'axes.titlesize': 'medium',
          'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'lines.linewidth': 4,
          'axes.facecolor': 'none'}

plt.rcParams.update(params)
plt.rcParams.update({'font.size': 22})
plt.rcParams["font.weight"] = "bold"
plt.rcParams['axes.linewidth'] = 4

def getdata():
    data = pd.read_csv("feature_vectors.csv").to_numpy()
    labels = pd.read_csv("diagnosis_label.csv").to_numpy().squeeze()
    labelscenter = pd.read_csv("center_label.csv").to_numpy().squeeze()

    labelsfinal = labels + labelscenter

    # better way to perform this
    labelsdiseases1 = np.where(labelsfinal == 'HCUMCG')
    centerlabels1 = labelsfinal[labelsdiseases1]
    labelsdiseases2 = np.where(labelsfinal == 'HCUGOSM')
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
    ######UMCG indices##########

    # ADindices = np.where(labels=='AD')

    UMCGindices = np.where(labelscenter == 'UMCG')

    # Remove UMCH 1st from data, centerlabels and diseaselabels

    datanoUMCG = np.delete(data, UMCGindices, axis=0)

    labelsnoUMCG = np.delete(labels, UMCGindices, axis=0)

    labelscenternoUMCG = np.delete(labelscenter, UMCGindices, axis=0)

    ############################################

    ### get AD indices

    ADindices = np.where(labelsnoUMCG == 'AD')

    # datacenter is the data with no umcg and AD incides

    finaldatacenter = np.delete(datanoUMCG, ADindices, axis=0)

    # Final center labels with no AD and UMCG
    finalcenterlabels = np.delete(labelscenternoUMCG, ADindices, axis=0)

    # labelsnoAD for disease data
    labelsnoAD = np.delete(labelsnoUMCG, ADindices, axis=0)

    PDindices = np.where(labelsnoAD == 'PD')

    #################  FINAL DATA CENTER 1 is FINAL###################

    final_center_data = np.delete(finaldatacenter, PDindices, axis=0)

    final_center_labels = np.delete(finalcenterlabels, PDindices, axis=0)

    ###############################################

    final_data_disease = finaldatacenter

    final_labels_disease = labelsnoAD + finalcenterlabels

    HC_UGOSM_Index = np.where(final_labels_disease == 'HCUGOSM')
    HC_CUN_Index = np.where(final_labels_disease == 'HCCUN')
    final_labels_disease[HC_UGOSM_Index] = 'HC'
    final_labels_disease[HC_CUN_Index] = 'HC'

    HCindices = HC_UGOSM_Index + HC_CUN_Index

    return final_center_data, final_center_labels, final_data_disease, final_labels_disease, HCindices


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


def model_definition_disease(correctionmatrix, logger):
    model1 = GMLVQ(
        distance_type="adaptive-squared-euclidean",
        activation_type="sigmoid",
        activation_params={"beta": 2},
        solver_type="wgd",
        solver_params={"max_runs": 50, "step_size": np.array([0.06, 0.03]), "callback": logger},
        # solver_params={"max_runs": 10,"batch_size":1,"step_size": np.array([0.1, 0.05])},
        random_state=1428,
        relevance_correction=correctionmatrix
    )
    return model1


def not_sampled(scalar, data, labels, correctionmatrix, disease):
    logger = ProcessLogger()
    if disease:
        model = model_definition_disease(correctionmatrix, logger)
    else:
        model = model_definition_center(logger)

    pipeline = make_pipeline(scalar, model)
    pipeline.fit(data, labels)
    pipeline.predict(data)
    return pipeline


def sampled(scalar, data, labels, correctionmatrix, disease):
    logger = ProcessLogger()
    if disease:
        model = model_definition_disease(correctionmatrix, logger)
    else:
        model = model_definition_center(logger)

    oversample = RandomOverSampler(sampling_strategy='minority')
    data, labels = oversample.fit_resample(data, labels)

    pipeline = make_pipeline(scalar, model)
    pipeline.fit(data, labels)
    pipeline.predict(data)
    return pipeline,data, labels


def train_modelkfold(data, label, disease, correctionmatrix, repeated, scalar, folds):
    modelmatrix = np.zeros((repeated, folds), dtype=object)
    train_dataM = np.zeros((repeated, folds), dtype=object)
    accuracies = np.zeros((repeated, folds), dtype=object)
    train_labelsM = np.zeros((repeated, folds), dtype=object)
    testlabelsM = np.zeros((repeated, folds), dtype=object)
    predictedM = np.zeros((repeated, folds), dtype=object)
    probablitiesM = np.zeros((repeated, folds), dtype=object)
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
            trainX, trainY, testX, testY = data[training_indices], label[training_indices], data[testing_indices], \
                                           label[testing_indices]

            pipeline = not_sampled(scalar, trainX, trainY, correctionmatrix, disease)
            predicted = pipeline.predict(testX)

            accuracy = 0
            correct = 0

            ##############################assigning to respectives matrices############################
            # ask about the fitted model
            modelmatrix[repeated, k] = pipeline[1]
            # stroing z transfomred data. CAn choose to store no z transfomrd data as well
            #train_dataM[repeated, k] = pipeline.transform(trainX,True)
            train_dataM[repeated, k] = trainX
            train_labelsM[repeated, k] = trainY
            traning_indicesM[repeated, k] = training_indices

            probabilties = pipeline.predict_proba(testX)

            probablitiesM[repeated, k] = probabilties

            testlabelsM[repeated, k] = testY

            predictedM[repeated, k] = predicted

            for i in range(len(predicted)):
                if(predicted[i]==testY[i]):
                    correct = correct+1
            accuracy = correct / len(testY)

            accuracies[repeated, k] = accuracy

            testing_indicesM[repeated, k] = testing_indices

    return modelmatrix, train_dataM, train_labelsM, testlabelsM, predictedM, probablitiesM, testing_indicesM, traning_indicesM,accuracies


def eigendecomposition(average_lambda):
    eigenvalues, eigenvectors = np.linalg.eigh(average_lambda)
    eigenvalues = np.flip(eigenvalues)
    eigenvectors = np.flip(eigenvectors, axis=1).T

    return eigenvalues, eigenvectors


def correction_matrix(eigvectorscenter,dimension,leading_eigenvectors):
    I = np.identity(dimension)
    outerproduct = np.zeros((dimension, dimension))
    for i in range(leading_eigenvectors):
        outerproduct += np.outer(eigvectorscenter.T[:, i], eigvectorscenter[i, :])
    correctionmatrix = I - outerproduct
    return correctionmatrix


def calculate_prototype(modelmatrix, repeated, dimension, scalarmodel):
    numberofprototypes = len(modelmatrix[0][0].prototypes_)
    prototypeaverage = np.zeros((numberofprototypes, dimension), dtype='float')
    modellist = list(chain.from_iterable(zip(*modelmatrix)))
    scalarlist = list(chain.from_iterable(zip(*scalarmodel)))
    for modelin in range(len(modellist)):
        for i in range(numberofprototypes):
            prototypeaverage[i] = np.add(prototypeaverage[i],
                                         scalarlist[modelin].inverse_transform(modellist[modelin].prototypes_[i]))
    return prototypeaverage / len(modellist)


def average_lambda(modelmatrix, dimension):
    modellist = list(chain.from_iterable(zip(*modelmatrix)))
    print(len(modellist))
    for i in range(len(modellist)):
        if i == 0:
            data = modellist[i].lambda_
        else:
            data = np.vstack((modellist[i].lambda_, data))
    data = data.reshape(len(modellist), dimension, dimension)
    mean = np.mean(data, axis=0)
    return mean


def average_lambda_diagonal(modelmatrix):
    modellist = list(chain.from_iterable(zip(*modelmatrix)))
    for i in range(len(modellist)):
        if i == 0:
            data = np.diagonal(modellist[i].lambda_)
        else:
            # data for i = 0 will be stacked at the bottom for the last iteration
            data = np.vstack((np.diagonal(modellist[i].lambda_), data))

    avg = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    return avg, std


def check_orthogonality(centermodel, diseasemodel,eigenvectorscenter,leading_eigenvectors):
    centerlist = list(chain.from_iterable(zip(*centermodel)))
    disease_center = list()
    disease_averagecenter = list()
    for j in range(leading_eigenvectors):
        for array, modeld in enumerate(diseasemodel[j]):
            disease_averagecenter.append(np.dot(modeld.eigenvectors_[:leading_eigenvectors, :], eigenvectorscenter[:leading_eigenvectors, :].T))
            for modelc in centerlist:
                # eigenvectors are row vectors
                disease_center.append((np.dot(modeld.eigenvectors_[:leading_eigenvectors, :], modelc.eigenvectors_[:leading_eigenvectors, :].T)))
    return disease_averagecenter, disease_center


def transform1(X, eigenvaluesaverage, eigenvectoraverage, scale):
    X = check_array(X)
    eigenvaluesaverage[eigenvaluesaverage < 0] = 0
    eigenvectoraverage_scaled = np.sqrt(eigenvaluesaverage[:, None]) * eigenvectoraverage
    if scale:
        return np.matmul(X, eigenvectoraverage_scaled.T)
    return np.matmul(X, eigenvectoraverage.T)


def center(center_data,center_labels,disease,correctionmatrix,repeated,scalar,folds,dimension,leading_eigenvectors):
    centermodel, trainc_data, trainc_labels, testlabelsc, predictedc, probabiltiesc, testing_indicesC, training_indicesC,accuracies= train_modelkfold(center_data,center_labels,disease,correctionmatrix,repeated,scalar,folds)
    average_lambda_center = average_lambda(centermodel,dimension)
    avgc, stdc = average_lambda_diagonal(centermodel)
    eigenvaluescenter,eigenvectorscenter = eigendecomposition(average_lambda_center)
    correctionmatrix_return = correction_matrix(eigenvectorscenter,dimension,leading_eigenvectors)
    return centermodel,testlabelsc, predictedc, probabiltiesc,average_lambda_center,avgc, stdc,eigenvaluescenter,eigenvectorscenter,correctionmatrix_return,accuracies


def disease_function(disease_data,disease_labels,disease,correctionmatrix,repeated,scalar,folds,dimension):
    diseasemodel, train_dataD, train_labelsD, testlabelsd, predictedd, probablitiesd, testing_indicesD, traning_indicesD,accuracies =  train_modelkfold(disease_data,disease_labels,disease,correctionmatrix,repeated,scalar,folds)
    average_lambda_disease = average_lambda(diseasemodel,dimension)
    avgd, stdd = average_lambda_diagonal(diseasemodel)
    eigenvaluesdisease,eigenvectorsdisease = eigendecomposition(average_lambda_disease)
    return diseasemodel,testlabelsd, predictedd, probablitiesd,average_lambda_disease,avgd, stdd,eigenvaluesdisease, eigenvectorsdisease,accuracies





def confusionmatrixc(test, predicted):
    CUNCUN, CUNUGOSM, UGOSMCUN, UGOSMUGOSM = confusion_matrix(test, predicted, labels=['CUN', 'UGOSM']).ravel()

    return CUNCUN, CUNUGOSM, UGOSMCUN, UGOSMUGOSM


def confusionmatrix_61classc(testlabelmatrix, predictedlabelmatrix):
    testlabelsdsinglec = np.concatenate(list(chain.from_iterable(zip(*testlabelmatrix))), axis=0)
    predicteddsinglec = np.concatenate(list(chain.from_iterable(zip(*predictedlabelmatrix))), axis=0)

    np.seterr(divide='ignore', invalid='ignore')

    # CUN_tpr = dict()
    # CUN_fpr = dict()
    #
    # UGOSM_tpr = dict()
    # UGOSM_fpr = dict()

    CUNCUN, CUNUGOSM, UGOSMCUN, UGOSMUGOSM = confusionmatrixc(testlabelsdsinglec, predicteddsinglec)

    nCUN_fpr = (UGOSMCUN) / (UGOSMCUN + UGOSMUGOSM)
    nCUN_tpr = (CUNCUN) / (CUNCUN + CUNUGOSM)

    nUGOSM_fpr = (CUNUGOSM) / (CUNUGOSM + CUNCUN)
    nUGOSM_tpr = (UGOSMUGOSM) / (UGOSMUGOSM + UGOSMCUN)

    #############################
    CUN_fpr = nCUN_fpr
    CUN_tpr = nCUN_tpr

    UGOSM_fpr = nUGOSM_fpr
    UGOSM_tpr = nUGOSM_tpr

    return CUN_fpr, CUN_tpr, UGOSM_fpr, UGOSM_tpr


def confusionmatrix_6classc(testlabelmatrix, predictedlabelmatrix):
    testlist = list(chain.from_iterable(zip(*testlabelmatrix)))
    predictedlist = list(chain.from_iterable(zip(*predictedlabelmatrix)))
    # storing the values
    np.seterr(divide='ignore', invalid='ignore')

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


def center_dict(CUN_fpr, CUN_tpr, UGOSM_fpr, UGOSM_tpr, CUN_fpr1, CUN_tpr1, UGOSM_fpr1, UGOSM_tpr1, repeated, folds):
    n_classes = ['CUN', 'UGOSM']
    # plotdict_tpr1 = dict()
    # plotdict_fpr1 = dict()

    plotdict_tpr1 = CUN_tpr1, UGOSM_tpr1
    plotdict_fpr1 = CUN_fpr1, UGOSM_fpr1

    n_classes = ['CUN', 'UGOSM']
    plotdict_tpr = dict()
    plotdict_fpr = dict()

    for j in range(repeated * folds):
        plotdict_tpr[j] = CUN_tpr[j], UGOSM_tpr[j]
        plotdict_fpr[j] = CUN_fpr[j], UGOSM_fpr[j]
    return plotdict_tpr1, plotdict_fpr1, plotdict_tpr, plotdict_fpr


def plot_roc_center(testlabels, probabilties, plotdict_tpr1, plotdict_fpr1, plotdict_tpr, plotdict_fpr):
    testlist = list(chain.from_iterable(zip(*testlabels)))
    probabiltylist = list(chain.from_iterable(zip(*probabilties)))


    testlist_average = np.concatenate(list(chain.from_iterable(zip(*testlabels))), axis=0)
    probabiltylist_average = np.concatenate(list(chain.from_iterable(zip(*probabilties))), axis=0)
    fpru = dict()
    tpru = dict()
    roc_aucu = dict()
    for k in range(len(testlist)):

        n_classes = ['CUN', 'UGOSM']

        # structures
        fpr = dict()

        tpr = dict()

        roc_auc = dict()

        fpr1 = dict()

        tpr1 = dict()

        roc_auc1 = dict()

        colorrange = ['lightblue', 'lightblue']
        colorrange2 = ['red', 'green']
        for i in range(len(n_classes)):
            fpr[i], tpr[i], _ = roc_curve(testlist[k], probabiltylist[k][:, i], pos_label=n_classes[i],
                                          drop_intermediate=True)
            roc_auc[i] = auc(fpr[i], tpr[i])
            fpru[k] = fpr
            tpru[k] = tpr
            roc_aucu[k] = roc_auc

    for j in range(len(n_classes)):
        fpr1[j], tpr1[j], _ = roc_curve(testlist_average, probabiltylist_average[:, j], pos_label=n_classes[j], drop_intermediate=True)
        roc_auc1[j] = auc(fpr1[j], tpr1[j])

    # roc for each class
    fig, ax = plt.subplots()

    ax.plot([0, 1], [0, 1], 'k--')
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')

    for k1 in range(len(testlist)):
        for i1 in range(len(n_classes)):
            ax.plot(fpru[k1][i1], tpru[k1][i1], label='_nolegend_', color=colorrange[i1])

            # ax.plot(PDCUN_fpr[i], PDCUN_fpr[i],  marker="o", markersize="12")
            ax.plot(plotdict_fpr[k1][i1], plotdict_tpr[k1][i1], color=colorrange[i1])  # ,marker="o", markersize="12")

    # plotdict_tpr1 = ADUGOSM_tpr1,ADUMCG_tpr1,PDCUN_tpr1,PDUGOSM_tpr1,PDUMCG_tpr1
    # plotdict_fpr1 = ADUGOSM_fpr1,ADUMCG_fpr1,PDCUN_tpr1,PDUGOSM_fpr1,PDUMCG_fpr1

    # for i2 in range(len(n_classes)):
    #    ax.plot(fpr1[i2], tpr1[1], label='ROC curve (AUC = %0.2f) for %s' % (roc_auc1[1], n_classes[1]),color=colorrange2[i2])
    #    ax.plot(plotdict_fpr1c[1], plotdict_tpr1[1], color=colorrange2[1],marker="o", markersize="12")

    ax.plot(fpr1[1], tpr1[1], label='ROC curve (AUC = %0.2f) for %s' % (roc_auc1[1], n_classes[1]),
            color=colorrange2[1])
    ax.plot(plotdict_fpr1[1], plotdict_tpr1[1], color=colorrange2[1], marker="o", markersize="12")

    ax.legend(loc="right")
    ax.grid(False)
    fig.savefig('roc6class.png')
    # ax.grid(alpha=.4)
    plt.show()


def confusionmatrixd(test, predicted):
    HCHC,HCDCUN,HCPDUGOSM,PDCUNHC,PDCUNPDCUN,PDCUNPDUGOSM,PDUGOSMHC,PDUGOSMPDCUN,PDUGOSMPDUGOSM = confusion_matrix(test, predicted, labels=['HC','PDCUN', 'PDUGOSM']).ravel()
    return HCHC,HCDCUN,HCPDUGOSM,PDCUNHC,PDCUNPDCUN,PDCUNPDUGOSM,PDUGOSMHC,PDUGOSMPDCUN,PDUGOSMPDUGOSM


def confusionmatrix_61class_d(testlabelmatrix, predictedlabelmatrix):
    testlabelsdsinglec = np.concatenate(list(chain.from_iterable(zip(*testlabelmatrix))), axis=0)
    predicteddsinglec = np.concatenate(list(chain.from_iterable(zip(*predictedlabelmatrix))), axis=0)
    np.seterr(divide='ignore', invalid='ignore')



    HCHC,HCPDCUN,HCPDUGOSM,PDCUNHC,PDCUNPDCUN,PDCUNPDUGOSM,PDUGOSMHC,PDUGOSMPDCUN,PDUGOSMPDUGOSM  = confusionmatrixd(testlabelsdsinglec,predicteddsinglec)

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


    return HC_fpr,HC_tpr,PDCUN_fpr,PDCUN_tpr,PDUGOSM_fpr,PDUGOSM_tpr

def confusionmatrix_6class_d(testlabelmatrix, predictedlabelmatrix):
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

    return HC_fpr,HC_tpr,PDCUN_fpr,PDCUN_tpr,PDUGOSM_fpr,PDUGOSM_tpr

def disease_dict(HC_fpr,HC_tpr,PDCUN_fpr,PDCUN_tpr,PDUGOSM_fpr,PDUGOSM_tpr, HC_fpr1,HC_tpr1,PDCUN_fpr1,PDCUN_tpr1,PDUGOSM_fpr1,PDUGOSM_tpr1, repeated, folds):
    n_classes=['HC','PDCUN', 'PDUGOSM']

    plotdict_tpr1 = HC_tpr1, PDCUN_tpr1, PDUGOSM_tpr1
    plotdict_fpr1 = HC_fpr1, PDCUN_fpr1, PDUGOSM_fpr1

    plotdict_tpr = dict()
    plotdict_fpr = dict()

    # for j in range(len(repeated)*folds):
    for j in range(repeated * folds):
        plotdict_tpr[j] = HC_tpr[j], PDCUN_tpr[j], PDUGOSM_tpr[j]
        plotdict_fpr[j] = HC_fpr[j], PDCUN_fpr[j], PDUGOSM_fpr[j]

    return plotdict_tpr1, plotdict_fpr1, plotdict_tpr, plotdict_fpr


def plot_roc_disease(testlabels, probabilties,plotdict_tpr1, plotdict_fpr1,plotdict_tpr, plotdict_fpr):
    testlist = list(chain.from_iterable(zip(*testlabels)))
    probabiltylist = list(chain.from_iterable(zip(*probabilties)))

    testlist_average = np.concatenate(list(chain.from_iterable(zip(*testlabels))), axis=0)
    probabiltylist_average = np.concatenate(list(chain.from_iterable(zip(*probabilties))), axis=0)
    fpru = dict()
    tpru = dict()
    roc_aucu = dict()

    n_classes = ['HC', 'PDCUN', 'PDUGOSM']

    # structures
    fpr = dict()

    tpr = dict()

    roc_auc = dict()

    fpr1 = dict()

    tpr1 = dict()

    roc_auc1 = dict()

    colorrange = ['green', 'red', 'blue']

    for k in range(len(testlist)):


        colorrange2 = ['green', 'red', 'blue']
        # for all the rocs from k fold
        for i in range(len(n_classes)):
            # dictionary fpr tpr and roc calcualted per class per model and then stored in a bigger dict
            # Dict of Dict
            fpr[i], tpr[i], _ = roc_curve(testlist[k], probabiltylist[k][:, i], pos_label=n_classes[i],
                                          drop_intermediate=True)
            roc_auc[i] = auc(fpr[i], tpr[i])
            fpru[k] = fpr
            tpru[k] = tpr
            roc_aucu[k] = roc_auc

    # One average ROC
    for j in range(len(n_classes)):
        fpr1[j], tpr1[j], _ = roc_curve(testlist_average, probabiltylist_average[:, j], pos_label=n_classes[j],
                                        drop_intermediate=True)
        roc_auc1[j] = auc(fpr1[j], tpr1[j])

    # roc for each class
    fig, ax = plt.subplots()

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')

    for i2 in range(len(n_classes)):
        ax.plot(fpr1[i2], tpr1[i2], label='ROC curve (AUC = %0.2f) for %s' % (roc_auc1[i2], n_classes[i2]),
                color=colorrange2[i2])
        ax.plot(plotdict_fpr1[i2], plotdict_tpr1[i2], color=colorrange2[i2], marker="o", markersize="12")

    ax.legend(loc="right")
    ax.grid(False)
    fig.savefig('roc6disease.png')
    # ax.grid(alpha=.4)
    plt.show()
    return roc_aucu


def analytical_average_auc(roc_aucu, n_classes):
    average_ruc = dict()
    for i in range(len(n_classes)):
        sum = 0
        for key in roc_aucu:
            sum = np.add(sum, roc_aucu[key][i])
        average_ruc[i] = sum / len(roc_aucu)
    return average_ruc


# single model. For average you can simply do average prototype and plot with average eigen values and eigen vectors
def visualizeSinglemodeld(pipeline,model, data, labels,typeofdata,ymin,ymax, HCindices):
    data = pipeline[0].fit_transform(data)
    transformed_data = model.transform(data, True)

    # all examples 1st and 2nd feature
    x_d = transformed_data[:, 0]
    y_d = transformed_data[:, 1]

    transformed_model = transform1(model.prototypes_, pipeline[1].eigenvalues_, pipeline[1].eigenvectors_, True)
    x_m = transformed_model[:, 0]
    y_m = transformed_model[:, 1]

    fig, ax = plt.subplots()

    colors = ['lightgreen','red','orange']
    colorsHC = ['darkviolet','teal']
    # check the ordering from 4 class
    labelHC = ['HCUGOSM','HCCUN']
    labelss = ['HC','PDCUN_late','PDUGOSM_early']


    # check the ordering from 4 class
    plt.rcParams['lines.solid_capstyle'] = 'round'

    for i, cls in enumerate(model.classes_):
        ii = cls == labels
        if(i == 0):
            for hc in range(len(HCindices)):
                ax.scatter(
                x_d[HCindices[hc]],
                y_d[HCindices[hc]],
                c="white",
                s=200,
                alpha=0.37,
                edgecolors=colorsHC[hc],
                linewidth=2.5,
                label = labelHC[hc])
        else:
            ax.scatter(
                x_d[ii],
                y_d[ii],
                c="white",
                s=200,
                alpha=0.37,
                edgecolors=colors[i],
                linewidth=2.5,
                label = labelss[i])


    for i, txt in enumerate(labelss):
        ax.annotate(labelss[i], (x_m[i], y_m[i]))
    ax.scatter(x_m, y_m, c=colors, s=500, alpha=0.8, edgecolors="black")
    #c then d
    ax.set_ylim(ymin[3], ymax[3])
    ax.set_xlabel("Projection on the 1st Eigenvector of Λ")
    ax.set_ylabel("Projection on the 2nd Eigenvector of Λ")
    ax.legend()
    ax.grid(True)
    fig.savefig('disease_single_model' + typeofdata + '.png')



# removing the z transform
def visualizeSinglemodelc(pipeline,model, data, labels,typeofdata):

    data = pipeline[0].fit_transform(data)
    transformed_data = model.transform(data, True)
    # all examples 1st and 2nd feature
    x_d = transformed_data[:, 0]
    y_d = transformed_data[:, 1]



    x_d = transformed_data[:, 0]
    y_d = transformed_data[:, 1]

    transformed_model = model.transform(model.prototypes_, scale=True)


    labelss = ['CUN', 'UGOSM']

    x_m = transformed_model[:, 0]
    y_m = transformed_model[:, 1]

    fig, ax = plt.subplots()
    colors = ['tomato', 'olive']
    for i, cls in enumerate(model.classes_):
        ii = cls == labels
        ax.scatter(
            x_d[ii],
            y_d[ii],
            c="none",
            s=200,
            alpha=0.37,
            edgecolors=colors[i],
            linewidth=2.5,
            label=model.classes_[model.prototypes_labels_[i]],
        )
    # probably code later to get these indices and plot
    for i, txt in enumerate(labelss):
        ax.annotate(labelss[i], (x_m[i], y_m[i]))
    ax.scatter(x_m, y_m, c=colors, s=500, alpha=0.8, edgecolors="black")
    ax.set_xlabel("Projection on the 1st Eigenvector of  Λ")
    ax.set_ylabel("Projection on the 2nd Eigenvector of Λ")
    ax.legend()
    ax.grid(True)

    fig.savefig('center_singlemodel' + typeofdata + '.png')



def plot_confusionmatrix_centre(testlabels, predicted, accuracieslist):
    accuracies_one = list(chain.from_iterable(zip(*accuracieslist)))
    print(sum(accuracies_one) / len(accuracies_one))

    test_list = np.concatenate(list(chain.from_iterable(zip(*testlabels))), axis=0)
    predicted_list = np.concatenate(list(chain.from_iterable(zip(*predicted))), axis=0)

    CUNCUN, CUNUGOSM, UGOSMCUN, UGOSMUGOSM = confusionmatrixc(test_list,
                                                              predicted_list)

    cmc_c = np.array([[CUNCUN, CUNUGOSM],
                      [UGOSMCUN, UGOSMUGOSM]])

    cmc_c = cmc_c.astype('float') / cmc_c.sum(axis=1)[:, np.newaxis]

    # Transform to df for easier plotting
    cm_df = pd.DataFrame(cmc_c, index=['CUN', 'UGOSM'], columns=['CUN', 'UGOSM'])

    plt.figure()
    sns.heatmap(cm_df, fmt='.2%', annot=True, cmap='Blues', cbar=False)

    plt.title('center data')
    plt.ylabel('True label')

    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix_toy_centre.png')
    plt.show()


# a function to plot the confusion matrix for centre
def plot_confusionmatrix_disease(testlabels, predicted, accuracieslist):
    accuracies_one = list(chain.from_iterable(zip(*accuracieslist)))
    print(sum(accuracies_one) / len(accuracies_one))

    test_list = np.concatenate(list(chain.from_iterable(zip(*testlabels))), axis=0)
    predicted_list = np.concatenate(list(chain.from_iterable(zip(*predicted))), axis=0)

    HCHC, HCDCUN, HCPDUGOSM, PDCUNHC, PDCUNPDCUN, PDCUNPDUGOSM, PDUGOSMHC, PDUGOSMPDCUN, PDUGOSMPDUGOSM = confusionmatrixd(
        test_list, predicted_list)

    cmc_d = np.array([[HCHC, HCDCUN, HCPDUGOSM],
                      [PDCUNHC, PDCUNPDCUN, PDCUNPDUGOSM],
                      [PDUGOSMHC, PDUGOSMPDCUN, PDUGOSMPDUGOSM]])

    cmc_d = cmc_d.astype('float') / cmc_d.sum(axis=1)[:, np.newaxis]

    cm_df = pd.DataFrame(cmc_d, index=['HC', 'PDCUN (L)', 'PDUGOSM (E)'], columns=['HC', 'PDCUN (L)', 'PDUGOSM (E)'])

    plt.figure()
    sns.heatmap(cm_df, fmt='.2%', annot=True, cmap='Blues', cbar=False)

    plt.title('Disease data')
    plt.ylabel('True label')

    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix_toy_disease.png')
    plt.show()


def ploteigenvalues(eigenvalues, eigenvectors, feature_names, averagelambda, std, d, typeofdata, ymin, ymax):



    fig, ax = plt.subplots()
    ax.bar(range(0, len(eigenvalues)), eigenvalues)
    if d == "disease":
        for i in range(3):
            plt.text(i , eigenvalues[i],  "{:.2f}".format(eigenvalues[i]),fontsize=14)
    else:
        plt.text(0, eigenvalues[0], "{:.2f}".format(eigenvalues[0]),fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Eigenvalue")
    ax.set_xlabel("Feature")
    ax.axhline(y=eigenvalues[0],color ='green', lw = 2)
    ax.grid(False)
    fig.savefig(d + 'eigenvalues' + typeofdata +'.png')

    fig, ax = plt.subplots()
    ax.bar(feature_names, eigenvectors[0, :])
    for i, v in enumerate(np.linspace(0, 35, 35)):
        plt.text(i, eigenvectors[0, i], "{:.2f}".format(eigenvectors[0, i]), color='k', fontsize=14)
    ax.set_ylim(ymin[0], ymax[0])
    ax.axhline(0, color='k')
    ax.set_ylabel("Weight")
    ax.set_xlabel("Feature")
    ax.grid(False)
    fig.savefig(d + 'Firsteigenvector' + typeofdata + '.png')


    fig, ax = plt.subplots()
    ax.bar(feature_names, eigenvectors[1, :])
    for i, v in enumerate(np.linspace(0, 35, 35)):
        plt.text(i, eigenvectors[1, i], "{:.2f}".format(eigenvectors[1, i]), color='k', fontsize=14)
    ax.set_ylim(ymin[1], ymax[1])
    ax.axhline(0,color='k')
    ax.set_ylabel("Weight")
    ax.set_xlabel("Feature")
    ax.grid(False)
    fig.savefig(d + 'Secondeigen_vector_without' + typeofdata + '.png')


    fig, ax = plt.subplots()
    ax.bar(feature_names, averagelambda, yerr=std,ecolor='dimgray')
    for i, v in enumerate(np.linspace(0, 35, 35)):
        plt.text(i, averagelambda[i] , "{:.2f}".format(averagelambda[i]),
                 ha='center',
                 va="bottom", color='forestgreen', fontsize=15)
    ax.set_ylim(ymin[2], ymax[2])
    ax.set_ylabel("Relevance")
    ax.set_xlabel("Diagonal elements of Λ")
    ax.grid(False)
    fig.savefig(d + 'RM_with' + typeofdata + '.png')




# Plot the eigenvalues of the eigenvectors of the relevance matrix.
def ploteigenvalueswithout(eigenvalues, eigenvectors, feature_names, averagelambda,std,d, typeofdata, ymin, ymax):
    fig, ax = plt.subplots()
    ax.bar(range(0, len(eigenvalues)), eigenvalues)
    if d == "disease":
        for i in range(3):
            plt.text(i, eigenvalues[i],  "{:.2f}".format(eigenvalues[i]), fontsize=14)
    else:
        plt.text(0, eigenvalues[0], "{:.2f}".format(eigenvalues[0]), fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Eigenvalue")
    ax.set_xlabel("Feature")
    ax.axhline(y=eigenvalues[0],color ='red', lw = 2)
    ax.grid(False)
    fig.savefig( d + 'Eigenvalues_without' + typeofdata + '.png')

    # Plot the first two eigenvectors of the relevance matrix, which  is called `omega_hat`.
    fig, ax = plt.subplots()
    ax.bar(feature_names, eigenvectors[0, :])
    for i, v in enumerate(np.linspace(0, 35, 35)):
        plt.text(i, eigenvectors[0, i], "{:.2f}".format(eigenvectors[0, i]), color='k', fontsize=14)
    ax.axhline(0, color='k')
    ax.set_ylim(ymin[0], ymax[0])
    ax.set_ylabel("Weight")
    ax.set_xlabel("Feature")
    ax.grid(False)
    fig.savefig( d + 'Firsteigen_vector_without'+ typeofdata + '.png')

    fig, ax = plt.subplots()
    ax.bar(feature_names, eigenvectors[1, :])
    for i, v in enumerate(np.linspace(0, 35, 35)):
        plt.text(i, eigenvectors[1, i], "{:.2f}".format(eigenvectors[1, i]), color='k', fontsize=14)
    ax.axhline(0, color='k')
    ax.set_ylim(ymin[1], ymax[1])
    ax.set_ylabel("Weight")
    ax.set_xlabel("Feature")
    ax.grid(False)
    fig.savefig(d + 'Secondeigen_vector_without' + typeofdata + '.png')



    fig, ax = plt.subplots()
    ax.bar(feature_names, averagelambda, yerr=std)
    for i, v in enumerate(np.linspace(0,35,35)):
        plt.text(i, averagelambda[i] + 0.000002, "{:.2f}".format(averagelambda[i]),
                 ha='center',
                 va="bottom", color='orangered', fontsize=14)
    ax.set_ylim(ymin[2], ymax[2])
    ax.set_ylabel("Relevance")
    ax.set_xlabel("Diagonal elements of Λ")
    ax.grid(False)
    fig.savefig(d + 'RM_without' + typeofdata + '.png')


def ploteigenvaluesingle(eigenvalues, eigenvectors, feature_names, averagelambda, d, typeofdata, ymin, ymax):
    fig, ax = plt.subplots()
    ax.bar(range(0, len(eigenvalues)), eigenvalues)
    if d == "disease":
        for i in range(1):
            plt.text(i, eigenvalues[i], "{:.2f}".format(eigenvalues[i]), fontsize=14)
    else:
        plt.text(0, eigenvalues[0], "{:.2f}".format(eigenvalues[0]), fontsize=14)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Eigenvalue")
    ax.set_xlabel("Feature")
    ax.axhline(y=eigenvalues[0], color='red', lw=2)
    ax.grid(False)
    fig.savefig(d + 'Eigenvalues_single' + typeofdata + '.png')

    fig, ax = plt.subplots()
    ax.bar(feature_names, eigenvectors[0, :])
    for i, v in enumerate(np.linspace(0, 35, 35)):
        plt.text(i, eigenvectors[0, i], "{:.2f}".format(eigenvectors[0, i]), color='k', fontsize=14)
    ax.axhline(0, color='k')
    ax.set_ylim(ymin[0], ymax[0])
    ax.set_ylabel("First Eigenvector weights")
    ax.set_xlabel("Feature")
    ax.grid(False)
    fig.savefig(d + 'Firsteigen_vector_single' + typeofdata + '.png')

    fig, ax = plt.subplots()
    ax.bar(feature_names, eigenvectors[1, :])
    for i, v in enumerate(np.linspace(0, 35, 35)):
        plt.text(i, eigenvectors[1, i], "{:.2f}".format(eigenvectors[1, i]), color='k', fontsize=14)
    ax.axhline(0, color='k')
    ax.set_ylim(ymin[1], ymax[1])
    ax.set_ylabel("Second Eigenvector weights")
    ax.set_xlabel("Feature")
    ax.grid(False)
    fig.savefig(d + 'Secondeigen_vector_single' + typeofdata + '.png')

    fig, ax = plt.subplots()
    ax.bar(feature_names, averagelambda)
    for i, v in enumerate(np.linspace(0, 35, 35)):
        plt.text(i, averagelambda[i], "{:.2f}".format(averagelambda[i]), color='k', fontsize=14)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Relevance")
    ax.set_xlabel("Diagonal elements of Λ")
    ax.grid(False)
    fig.savefig(d + 'RM_single' + typeofdata + '.png')
