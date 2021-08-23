# a lib file for 2 class early and late PD
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
from imblearn.over_sampling import SMOTE

matplotlib.rc("xtick", labelsize="small")
matplotlib.rc("ytick", labelsize="small")

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_array


params = {'legend.fontsize': 'medium',
          'figure.figsize': (23, 14),
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

    labelsdiseases1 = np.where(labelsfinal == 'HCUMCG')
    centerlabels1 = labelsfinal[labelsdiseases1]
    labelsdiseases2 = np.where(labelsfinal == 'HCUGOSM')
    centerlabels2 = labelsfinal[labelsdiseases2]
    labelsdiseases3 = np.where(labelsfinal == 'HCCUN')
    centerlabels3 = labelsfinal[labelsdiseases3]

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
    healthycontrolindices = np.where(diseaselabels_noAD == 'HC')

    final_data_disease = np.delete(diseasedata_noAD, healthycontrolindices, axis=0)

    diseaselabels_noHC = np.delete(diseaselabels_noAD, healthycontrolindices, axis=0)
    centerlabels_noHC = np.delete(centerlabels_noAD, healthycontrolindices, axis=0)
    final_labels_disease = diseaselabels_noHC + centerlabels_noHC

    return final_center_data, final_center_labels, final_data_disease, final_labels_disease


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
        random_state=1428)

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

            testing_indicesM[repeated, k] = testing_indices

    return modelmatrix, train_dataM, train_labelsM, testlabelsM, predictedM, probablitiesM, testing_indicesM, traning_indicesM


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
    centermodel, trainc_data, trainc_labels, testlabelsc, predictedc, probabiltiesc, testing_indicesC, training_indicesC = train_modelkfold(center_data,center_labels,disease,correctionmatrix,repeated,scalar,folds)
    average_lambda_center = average_lambda(centermodel,dimension)
    avgc, stdc = average_lambda_diagonal(centermodel)
    eigenvaluescenter,eigenvectorscenter = eigendecomposition(average_lambda_center)
    correctionmatrix_return = correction_matrix(eigenvectorscenter,dimension,leading_eigenvectors)
    return centermodel,testlabelsc, predictedc, probabiltiesc,average_lambda_center,avgc, stdc,eigenvaluescenter,eigenvectorscenter,correctionmatrix_return


def disease_function(disease_data,disease_labels,disease,correctionmatrix,repeated,scalar,folds,dimension):
    diseasemodel, train_dataD, train_labelsD, testlabelsd, predictedd, probablitiesd, testing_indicesD, traning_indicesD =  train_modelkfold(disease_data,disease_labels,disease,correctionmatrix,repeated,scalar,folds)
    average_lambda_disease = average_lambda(diseasemodel,dimension)
    avgd, stdd = average_lambda_diagonal(diseasemodel)
    eigenvaluesdisease,eigenvectorsdisease = eigendecomposition(average_lambda_disease)
    return diseasemodel,testlabelsd, predictedd, probablitiesd,average_lambda_disease,avgd, stdd,eigenvaluesdisease, eigenvectorsdisease



def confusionmatrixc(test, predicted):
    CUNCUN, CUNUGOSM, UGOSMCUN, UGOSMUGOSM = confusion_matrix(test, predicted, labels=['CUN', 'UGOSM']).ravel()

    return CUNCUN, CUNUGOSM, UGOSMCUN, UGOSMUGOSM


def confusionmatrix_61classc(testlabelmatrix, predictedlabelmatrix):
    testlabelsdsinglec = np.concatenate(list(chain.from_iterable(zip(*testlabelmatrix))), axis=0)
    predicteddsinglec = np.concatenate(list(chain.from_iterable(zip(*predictedlabelmatrix))), axis=0)

    np.seterr(divide='ignore', invalid='ignore')

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

    plotdict_tpr1 = CUN_tpr1, UGOSM_tpr1
    plotdict_fpr1 = CUN_fpr1, UGOSM_fpr1

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
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    # for k1 in range(len(testlist)):
    #     for i1 in range(len(n_classes)):
    #         ax.plot(fpru[k1][i1], tpru[k1][i1], label='_nolegend_', color=colorrange[i1])
    #         ax.plot(plotdict_fpr[k1][i1], plotdict_tpr[k1][i1], color=colorrange[i1])

    ax.plot(fpr1[1], tpr1[1], label='ROC curve (AUC = %0.2f) for %s' % (roc_auc1[1], n_classes[1]),
            color=colorrange2[1])
    ax.plot(plotdict_fpr1[1], plotdict_tpr1[1], color=colorrange2[1],markeredgewidth=1.0,markeredgecolor='k',marker="o", markersize="12")

    ax.legend(loc="best")
    ax.grid(False)
    fig.savefig('roc_centre.png')
    plt.show()


def confusionmatrixd(test, predicted):
    PDCUNPDCUN, PDCUNPDUGOSM, PDUGOSMPDCUN, PDUGOSMPDUGOSM = confusion_matrix(test, predicted,
                                                                              labels=['PDCUN', 'PDUGOSM']).ravel()
    return PDCUNPDCUN, PDCUNPDUGOSM, PDUGOSMPDCUN, PDUGOSMPDUGOSM

def confusionmatrix_61class_d(testlabelmatrix, predictedlabelmatrix):
    testlabelsdsinglec = np.concatenate(list(chain.from_iterable(zip(*testlabelmatrix))), axis=0)
    predicteddsinglec = np.concatenate(list(chain.from_iterable(zip(*predictedlabelmatrix))), axis=0)
    np.seterr(divide='ignore', invalid='ignore')


    PDCUNPDCUN,PDCUNPDUGOSM,PDUGOSMPDCUN,PDUGOSMPDUGOSM = confusionmatrixd(
        testlabelsdsinglec, predicteddsinglec)

    nPDCUN_fpr = (PDUGOSMPDCUN)/(PDUGOSMPDCUN+PDUGOSMPDUGOSM)
    nPDCUN_tpr = (PDCUNPDCUN)/(PDCUNPDCUN+PDCUNPDUGOSM)


    nPDUGOSM_fpr = (PDCUNPDUGOSM)/(PDCUNPDUGOSM+PDCUNPDCUN)
    nPDUGOSM_tpr = (PDUGOSMPDUGOSM)/(PDUGOSMPDUGOSM+PDUGOSMPDCUN)

    #############################
    PDCUN_fpr = nPDCUN_fpr
    PDCUN_tpr = nPDCUN_tpr

    PDUGOSM_fpr = nPDUGOSM_fpr
    PDUGOSM_tpr = nPDUGOSM_tpr

    return PDCUN_fpr,PDCUN_tpr,PDUGOSM_fpr,PDUGOSM_tpr


def confusionmatrix_6class_d(testlabelmatrix, predictedlabelmatrix):
    testlist = list(chain.from_iterable(zip(*testlabelmatrix)))
    predictedlist = list(chain.from_iterable(zip(*predictedlabelmatrix)))
    # storing the values
    np.seterr(divide='ignore', invalid='ignore')

    PDCUN_tpr = dict()
    PDCUN_fpr = dict()

    PDUGOSM_tpr = dict()
    PDUGOSM_fpr = dict()

    for i in range(len(testlist)):
        PDCUNPDCUN,PDCUNPDUGOSM,PDUGOSMPDCUN,PDUGOSMPDUGOSM = confusionmatrixd(
            testlist[i], predictedlist[i])

        nPDCUN_fpr = (PDUGOSMPDCUN)/(PDUGOSMPDCUN+PDUGOSMPDUGOSM)
        nPDCUN_tpr = (PDCUNPDCUN)/(PDCUNPDCUN+PDCUNPDUGOSM)


        nPDUGOSM_fpr = (PDCUNPDUGOSM)/(PDCUNPDUGOSM+PDCUNPDCUN)
        nPDUGOSM_tpr = (PDUGOSMPDUGOSM)/(PDUGOSMPDUGOSM+PDUGOSMPDCUN)

        #############################
        PDCUN_fpr[i] = nPDCUN_fpr
        PDCUN_tpr[i] = nPDCUN_tpr


        PDUGOSM_fpr[i] = nPDUGOSM_fpr
        PDUGOSM_tpr[i] = nPDUGOSM_tpr

    return PDCUN_fpr,PDCUN_tpr,PDUGOSM_fpr,PDUGOSM_tpr


def disease_dict(PDCUN_fpr, PDCUN_tpr, PDUGOSM_fpr, PDUGOSM_tpr, PDCUN_fpr1,
                 PDCUN_tpr1, PDUGOSM_fpr1, PDUGOSM_tpr1, repeated, folds):

    plotdict_tpr1 = PDCUN_tpr1, PDUGOSM_tpr1
    plotdict_fpr1 = PDCUN_fpr1, PDUGOSM_fpr1

    plotdict_tpr = dict()
    plotdict_fpr = dict()

    for j in range(repeated * folds):
        plotdict_tpr[j] = PDCUN_tpr[j], PDUGOSM_tpr[j]
        plotdict_fpr[j] = PDCUN_fpr[j], PDUGOSM_fpr[j]

    return plotdict_tpr1, plotdict_fpr1, plotdict_tpr, plotdict_fpr


def plot_roc_disease(testlabels, probabilties,plotdict_tpr1, plotdict_fpr1,plotdict_tpr, plotdict_fpr):
    testlist = list(chain.from_iterable(zip(*testlabels)))
    probabiltylist = list(chain.from_iterable(zip(*probabilties)))

    testlist_average = np.concatenate(list(chain.from_iterable(zip(*testlabels))), axis=0)
    probabiltylist_average = np.concatenate(list(chain.from_iterable(zip(*probabilties))), axis=0)
    fpru = dict()
    tpru = dict()
    roc_aucu = dict()
    for k in range(len(testlist)):

        n_classes = ['PDCUN', 'PDUGOSM']

        # structures
        fpr = dict()

        tpr = dict()

        roc_auc = dict()

        fpr1 = dict()

        tpr1 = dict()

        roc_auc1 = dict()

        colorrange = ['green', 'red']
        colorrange2 = ['lightgreen', 'pink']
        # for all the rocs from k fold
        for i in range(len(n_classes)):
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


    ax.plot(fpr1[1], tpr1[1], label='ROC curve (AUC = %0.2f) for %s' % (roc_auc1[1], n_classes[1]),
                color=colorrange2[1])
    ax.plot(plotdict_fpr1[1], plotdict_tpr1[1], color=colorrange2[1],markeredgewidth=1.0,markeredgecolor='k',marker="o", markersize="12")

    ax.legend(loc="best")
    ax.grid(False)
    fig.savefig('roc_disease.png')
    plt.show()


def analytical_average_auc(roc_aucu, n_classes):
    average_ruc = dict()
    for i in range(len(n_classes)):
        sum = 0
        for key in roc_aucu:
            sum = np.add(sum, roc_aucu[key][i])
        average_ruc[i] = sum / len(roc_aucu)
    return average_ruc


# single model. For average you can simply do average prototype and plot with average eigen values and eigen vectors
def visualizeSinglemodeld(pipeline,model, data, labels,typeofdata,ymin,ymax):

    data = pipeline[0].fit_transform(data)
    transformed_data = model.transform(data, True)

    # all examples 1st and 2nd feature
    x_d = transformed_data[:, 0]
    y_d = transformed_data[:, 1]

    transformed_model = transform1(model.prototypes_, pipeline[1].eigenvalues_, pipeline[1].eigenvectors_, True)
    #transformed_model = model.transform(model.prototypes_,True)
    x_m = transformed_model[:, 0]
    y_m = transformed_model[:, 1]

    fig, ax = plt.subplots()

    labelss = ['PDCUN_late', 'PDUGOSM_early']
    colors = ['deeppink', 'limegreen']

    # check the ordering from 4 class
    plt.rcParams['lines.solid_capstyle'] = 'round'

    for i, cls in enumerate(model.classes_):
        ii = cls == labels
        ax.scatter(
            x_d[ii],
            y_d[ii],
            c="white",
            s=200,
            alpha=0.37,
            edgecolors=colors[i],
            linewidth=2.5,
            label=labelss[i]
        )


    for i, txt in enumerate(labelss):
        ax.annotate(labelss[i], (x_m[i], y_m[i]))
    ax.scatter(x_m, y_m, c=colors, s=500, alpha=0.8, edgecolors="black")
    #c then d
    ax.set_ylim(ymin[3], ymax[3])
    ax.set_xlabel("First eigenvector")
    ax.set_ylabel("Second eigenvector")
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
    ax.set_xlabel("First eigenvector")
    ax.set_ylabel("Second eigenvector")
    ax.legend()
    ax.grid(True)
    print(model.classes_)
    fig.savefig('center_singlemodel' + typeofdata + '.png')
    print(model.classes_[model.prototypes_labels_[0]])
    print(model.classes_[model.prototypes_labels_[1]])


# Plot the eigenvalues of the eigenvectors of the relevance matrix.
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
    ax.set_xlabel("Feature")
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
    ax.bar(feature_names, np.diagonal(averagelambda), yerr=std)
    for i, v in enumerate(np.linspace(0,35,35)):
        plt.text(i, averagelambda[i] + 0.000002, "{:.2f}".format(averagelambda[i]),
                 ha='center',
                 va="bottom", color='orangered', fontsize=14)
    ax.set_ylim(ymin[2], ymax[2])
    ax.set_ylabel("Relevance")
    ax.set_xlabel("Feature")
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
    ax.set_xlabel("Feature")
    ax.grid(False)
    fig.savefig(d + 'RM_single' + typeofdata + '.png')