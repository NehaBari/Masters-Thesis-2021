import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklvq import GMLVQ
from itertools import chain
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, auc

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
    data = pd.read_csv("fourcl.csv").to_numpy()
    labels = pd.read_csv("fourcl_labels.csv").to_numpy().squeeze()

    labelscenter = labels
    labelscenter = np.where(labels == 'A1', 'A', labelscenter)
    labelscenter = np.where(labels == 'A2', 'A', labelscenter)
    labelscenter = np.where(labels == 'B1', 'B', labelscenter)
    labelscenter = np.where(labels == 'B2', 'B', labelscenter)

    # changing labels to Class A and B
    labelsdiseases = labels
    labelsdiseases = np.where(labels == 'A1', '1', labelsdiseases)
    labelsdiseases = np.where(labels == 'A2', '2', labelsdiseases)
    labelsdiseases = np.where(labels == 'B1', '1', labelsdiseases)
    labelsdiseases = np.where(labels == 'B2', '2', labelsdiseases)

    scalar = StandardScaler()

    return data, labels, labelscenter, labelsdiseases, scalar





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
    fitted_model = pipeline.fit(data, labels)
    #predicted = pipeline.predict(data)
    return pipeline


def sampled(scalar, data, labels, correctionmatrix, disease):
    logger = ProcessLogger()
    if disease:
        model = model_definition_disease(correctionmatrix, logger)
    else:
        model = model_definition_center(logger)

    oversample = RandomOverSampler(sampling_strategy='minority')
    data_sampled, labels_sampled = oversample.fit_resample(data, labels)

    pipeline = make_pipeline(scalar, model)
    fitted_model = pipeline.fit(data_sampled, labels_sampled)
    predicted = pipeline.predict(data)
    predicted = pipeline.predict(data_sampled)
    return pipeline, data, labels


def train_modelkfold(data, label, disease, correctionmatrix, repeated, scalar, folds):
    modelmatrix = np.zeros((repeated, folds), dtype=object)


    accuracies = np.zeros((repeated, folds), dtype=object)

    train_dataM = np.zeros((repeated, folds), dtype=object)
    ############################################################
    test_dataM = np.zeros((repeated, folds), dtype=object)
    ############################################################
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
            # train_dataM[repeated, k] = pipeline.transform(trainX,True)
            train_dataM[repeated, k] = trainX
            train_labelsM[repeated, k] = trainY
            traning_indicesM[repeated, k] = training_indices

            accuracy = 0
            correct = 0

            probabilties = pipeline.predict_proba(testX)

            probablitiesM[repeated, k] = probabilties

            test_dataM[repeated,k] = testX

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


def correction_matrix(eigvectorscenter, dimension, leading_eigenvectors):
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


def check_orthogonality(centermodel, diseasemodel, eigenvectorscenter, leading_eigenvectors):
    centerlist = list(chain.from_iterable(zip(*centermodel)))
    disease_center = list()
    disease_averagecenter = list()
    for j in range(leading_eigenvectors):
        for array, modeld in enumerate(diseasemodel[j]):
            disease_averagecenter.append(
                np.dot(modeld.eigenvectors_[:leading_eigenvectors, :], eigenvectorscenter[:leading_eigenvectors, :].T))
            for modelc in centerlist:
                # eigenvectors are row vectors
                disease_center.append((np.dot(modeld.eigenvectors_[:leading_eigenvectors, :],
                                              modelc.eigenvectors_[:leading_eigenvectors, :].T)))
    return disease_averagecenter, disease_center


def transform1(X, eigenvaluesaverage, eigenvectoraverage, scale):
    X = check_array(X)
    eigenvaluesaverage[eigenvaluesaverage < 0] = 0
    eigenvectoraverage_scaled = np.sqrt(eigenvaluesaverage[:, None]) * eigenvectoraverage
    if scale:
        return np.matmul(X, eigenvectoraverage_scaled.T)
    return np.matmul(X, eigenvectoraverage.T)


def center(center_data, center_labels, disease, correctionmatrix, repeated, scalar, folds, dimension,
           leading_eigenvectors):
    centermodel, trainc_data, trainc_labels, testlabelsc, predictedc, probabiltiesc, testing_indicesC, training_indicesC,accuracies = train_modelkfold(
        center_data, center_labels, disease, correctionmatrix, repeated, scalar, folds)
    average_lambda_center = average_lambda(centermodel, dimension)
    avgc, stdc = average_lambda_diagonal(centermodel)
    eigenvaluescenter, eigenvectorscenter = eigendecomposition(average_lambda_center)
    correctionmatrix_return = correction_matrix(eigenvectorscenter, dimension, leading_eigenvectors)
    return centermodel, testlabelsc, predictedc, probabiltiesc, average_lambda_center, avgc, stdc, eigenvaluescenter, eigenvectorscenter, correctionmatrix_return, accuracies

def disease_function(disease_data, disease_labels, disease, correctionmatrix, repeated, scalar, folds, dimension):
    diseasemodel, train_dataD, train_labelsD, testlabelsd, predictedd, probablitiesd, testing_indicesD, traning_indicesD ,accuracies= train_modelkfold(
        disease_data, disease_labels, disease, correctionmatrix, repeated, scalar, folds)
    average_lambda_disease = average_lambda(diseasemodel, dimension)
    avgd, stdd = average_lambda_diagonal(diseasemodel)
    eigenvaluesdisease, eigenvectorsdisease = eigendecomposition(average_lambda_disease)

    return diseasemodel, testlabelsd, predictedd, probablitiesd, average_lambda_disease, avgd, stdd, eigenvaluesdisease, eigenvectorsdisease, accuracies

def confusionmatrixc(test, predicted):
    oneone, onetwo, twoone, twotwo = confusion_matrix(test, predicted, labels=['A', 'B']).ravel()

    return oneone, onetwo, twoone, twotwo


def confusionmatrixd(test, predicted):
    oneone, onetwo, twoone, twotwo = confusion_matrix(test, predicted, labels=['1', '2']).ravel()

    return oneone, onetwo, twoone, twotwo


# two class and no use of classes explcitly so will stick to using only one for both disease and center
def confusionmatrix_61class(testlabelmatrix, predictedlabelmatrix, type):
    testlabelsdsingle = np.concatenate(list(chain.from_iterable(zip(*testlabelmatrix))), axis=0)
    predicteddsingle = np.concatenate(list(chain.from_iterable(zip(*predictedlabelmatrix))), axis=0)
    np.seterr(divide='ignore', invalid='ignore')

    if (type == 'center'):
        oneone, onetwo, twoone, twotwo = confusionmatrixc(testlabelsdsingle, predicteddsingle)
    else:
        oneone, onetwo, twoone, twotwo = confusionmatrixd(testlabelsdsingle, predicteddsingle)

    none_fpr = (twoone) / (twoone + twotwo)
    none_tpr = (oneone) / (oneone + onetwo)

    ntwo_fpr = (onetwo) / (onetwo + oneone)
    ntwo_tpr = (twotwo) / (twotwo + twoone)

    #############################
    one_fpr = none_fpr
    one_tpr = none_tpr

    two_fpr = ntwo_fpr
    two_tpr = ntwo_tpr

    return one_fpr, one_tpr, two_fpr, two_tpr


def confusionmatrix_6class(testlabelmatrix, predictedlabelmatrix, type):
    testlist = list(chain.from_iterable(zip(*testlabelmatrix)))
    predictedlist = list(chain.from_iterable(zip(*predictedlabelmatrix)))
    # storing the values
    np.seterr(divide='ignore', invalid='ignore')

    one_tpr = dict()
    one_fpr = dict()

    two_tpr = dict()
    two_fpr = dict()
    # all are averge rates
    # later might store in a matrix to index exact model
    for i in range(len(testlist)):

        if (type == 'center'):
            oneone, onetwo, twoone, twotwo = confusionmatrixc(testlist[i], predictedlist[i])
        else:
            oneone, onetwo, twoone, twotwo = confusionmatrixd(testlist[i], predictedlist[i])

        none_fpr = (twoone) / (twoone + twotwo)
        none_tpr = (oneone) / (oneone + onetwo)

        ntwo_fpr = (onetwo) / (onetwo + oneone)
        ntwo_tpr = (twotwo) / (twotwo + twoone)

        #############################
        one_fpr[i] = none_fpr
        one_tpr[i] = none_tpr

        two_fpr[i] = ntwo_fpr
        two_tpr[i] = ntwo_tpr

    return one_fpr, one_tpr, two_fpr, two_tpr


def dictionary(one_fpr, one_tpr, two_fpr, two_tpr, one_fpr1, one_tpr1, two_fpr1, two_tpr1, repeated, folds):
    plotdict_tpr1 = one_tpr1, two_tpr1
    plotdict_fpr1 = one_fpr1, two_fpr1

    plotdict_tpr = dict()
    plotdict_fpr = dict()

    for j in range(repeated * folds):
        plotdict_tpr[j] = one_tpr[j], two_tpr[j]
        plotdict_fpr[j] = one_fpr[j], two_fpr[j]
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

        n_classes = ['A', 'B']

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
        fpr1[j], tpr1[j], _ = roc_curve(testlist_average, probabiltylist_average[:, j], pos_label=n_classes[j],
                                        drop_intermediate=True)
        roc_auc1[j] = auc(fpr1[j], tpr1[j])

    # roc for each class
    fig, ax = plt.subplots()

    ax.plot([0, 1], [0, 1], 'k--')
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')


    # for k1 in range(len(testlist)):
    #     for i1 in range(1):
    #         ax.plot(fpru[k1][1], tpru[k1][1], label='_nolegend_', color=colorrange[1])
    #         ax.plot(plotdict_fpr[k1][1], plotdict_tpr[k1][1], color=colorrange[1])

    ax.plot(fpr1[1], tpr1[1], label='ROC curve (AUC = %0.2f) for %s' % (roc_auc1[1], n_classes[1]),
            color=colorrange2[1])
    ax.plot(plotdict_fpr1[1], plotdict_tpr1[1], color=colorrange2[1], markeredgewidth=1.0, markeredgecolor='k',
            marker="o", markersize="12")

    ax.legend(loc="right")
    ax.grid(False)
    fig.savefig('roctoy_center.png')
    # ax.grid(alpha=.4)
    plt.show()


def plot_roc_disease(testlabels, probabilties, plotdict_tpr1, plotdict_fpr1, plotdict_tpr, plotdict_fpr):
    testlist = list(chain.from_iterable(zip(*testlabels)))
    probabiltylist = list(chain.from_iterable(zip(*probabilties)))

    testlist_average = np.concatenate(list(chain.from_iterable(zip(*testlabels))), axis=0)
    probabiltylist_average = np.concatenate(list(chain.from_iterable(zip(*probabilties))), axis=0)
    fpru = dict()
    tpru = dict()
    roc_aucu = dict()
    for k in range(len(testlist)):

        n_classes = ['1', '2']

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



    ax.plot(fpr1[1], tpr1[1], label='ROC curve (AUC = %0.2f) for %s' % (roc_auc1[1], n_classes[1]),
            color=colorrange2[1])
    ax.plot(plotdict_fpr1[1], plotdict_tpr1[1], markeredgewidth=1.0, markeredgecolor='k', color=colorrange2[1],
            marker="o", markersize="12")

    ax.legend(loc="right")
    ax.grid(False)
    fig.savefig('roctoy_disease.png')
    plt.show()


def analytical_average_auc(roc_aucu, n_classes):
    # each validation set ROC check if equal to the final ROC
    average_ruc = dict()
    for i in range(len(n_classes)):
        sum = 0
        for key in roc_aucu:
            sum = np.add(sum, roc_aucu[key][i])
        average_ruc[i] = sum / len(roc_aucu)
    return average_ruc


# single model. For average you can simply do average prototype and plot with average eigen values and eigen vectors
def visualizeSinglemodeld(pipeline,model, data, labels, typeofdata):

    data = pipeline[0].fit_transform(data)
    transformed_data = model.transform(data, True)
    print(np.mean(data))
    print(np.std(data))
    # all examples 1st and 2nd feature
    x_d = transformed_data[:, 0]
    y_d = transformed_data[:, 1]

    transformed_model = model.transform(model.prototypes_, scale=True)
    x_m = transformed_model[:, 0]
    y_m = transformed_model[:, 1]
    fig, ax = plt.subplots()

    labelss = ['1', '2']
    colors = ['deeppink', 'limegreen']
    # check the ordering from 4 class
    plt.rcParams['lines.solid_capstyle'] = 'round'

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
            label=model.classes_[model.prototypes_labels_[i]]
        )
    for i, txt in enumerate(labelss):
        ax.annotate(labelss[i], (x_m[i], y_m[i]))
    ax.scatter(x_m, y_m, c=colors, s=500, alpha=0.8, edgecolors="black")
    ax.set_xlabel("Projection on the 1st Eigenvector of Λ")
    ax.set_ylabel("Projection on the 2nd Eigenvector of Λ")
    ax.legend()
    ax.grid(True)
    fig.savefig('disease_single_model' + typeofdata + '.png')



def visualizeSinglemodelc(pipeline,model, datac, labels, typeofdata):

    data2 = pipeline[0].fit_transform(datac)
    transformed_data = model.transform(data2,True)
    print(np.mean(data2))
    print(np.std(data2))
    # all examples 1st and 2nd feature
    #transformed_data = data1
    x_d = transformed_data[:, 0]
    y_d = transformed_data[:, 1]

    transformed_model = model.transform(model.prototypes_, scale=True)

    labelss = ['A', 'B']
    x_m = transformed_model[:, 0]
    y_m = transformed_model[:, 1]
    fig, ax = plt.subplots()

    colors = ['tomato', 'olive']  # ,'red','lightgreen']

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
            label=model.classes_[model.prototypes_labels_[i]]
        )

    for i, txt in enumerate(labelss):
        ax.annotate(labelss[i], (x_m[i], y_m[i]))
    ax.scatter(x_m, y_m, c=colors, s=500, alpha=0.8, edgecolors="black")

    # x_extraticks = np.linspace(np.min(datac[:, 0]), np.max(datac[:, 0]), 6)
    # ax.set_xticks(list(x_extraticks))
    ax.set_xlabel("Projection on the 1st Eigenvector of Λ")
    ax.set_ylabel("Projection on the 2nd Eigenvector of Λ")
    ax.legend()
    ax.grid(True)
    fig.savefig('center_singlemodel' + typeofdata + '.png')


# a function to plot the confusion matrix for centre
def plot_confusionmatrix_centre(testlabels, predicted, accuracieslist):
    accuracies_one = list(chain.from_iterable(zip(*accuracieslist)))
    print(sum(accuracies_one) / len(accuracies_one))

    test_list = np.concatenate(list(chain.from_iterable(zip(*testlabels))), axis=0)
    predicted_list = np.concatenate(list(chain.from_iterable(zip(*predicted))), axis=0)

    oneonec, onetwoc, twoonec, twotwoc = confusionmatrixc(test_list,
                                                          predicted_list)

    cmc_c = np.array([[oneonec, onetwoc],
                      [twoonec, twotwoc]])

    cmc_c = cmc_c.astype('float') / cmc_c.sum(axis=1)[:, np.newaxis]

    # Transform to df for easier plotting
    cm_df = pd.DataFrame(cmc_c, index=['A', 'B'], columns=['A', 'B'])

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

    oneonec, onetwoc, twoonec, twotwoc = confusionmatrixd(test_list,
                                                          predicted_list)

    cmc_d = np.array([[oneonec, onetwoc],
                      [twoonec, twotwoc]])

    cmc_d = cmc_d.astype('float') / cmc_d.sum(axis=1)[:, np.newaxis]

    # Transform to df for easier plotting
    cm_df = pd.DataFrame(cmc_d, index=['1', '2'], columns=['1', '2'])

    plt.figure()
    sns.heatmap(cm_df, fmt='.2%', annot=True, cmap='Blues', cbar=False)

    plt.title('Disease data')
    plt.ylabel('True label')

    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix_toy_disease.png')
    plt.show()




# Plot the eigenvalues of the eigenvectors of the relevance matrix.
def ploteigenvalues(eigenvalues, eigenvectors, feature_names, averagelambda,std,d, typeofdata,ymin, ymax):

    fig, ax = plt.subplots()
    ax.bar(range(0, len(eigenvalues)), eigenvalues)
    if d == "disease":
        for i in range(1):
            plt.text(i, eigenvalues[i],  "{:.2f}".format(eigenvalues[i]),fontsize=14)
    else:
        plt.text(0, eigenvalues[0], "{:.2f}".format(eigenvalues[0]),fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Eigenvalue")
    ax.set_xlabel("Feature")
    ax.axhline(y=eigenvalues[0],color ='red', lw = 2)
    ax.grid(False)
    fig.savefig( d + 'Eigenvalues_without' + typeofdata + '.png')

    # Plot the first two eigenvectors of the relevance matrix, which  is called `omega_hat`.
    fig, ax = plt.subplots()
    ax.bar(feature_names, eigenvectors[0, :])
    ax.set_ylim(ymin[0], ymax[0])
    ax.axhline(0, color='k')
    ax.set_ylabel("Weight")
    ax.set_xlabel("Feature")
    ax.grid(False)
    fig.savefig(d + 'Firsteigenvector_with' + typeofdata + '.png')

    fig, ax = plt.subplots()
    #fig.suptitle("Second Eigenvector with correction " + d + typeofdata + "data")
    ax.bar(feature_names, eigenvectors[1, :])
    ax.set_ylim(ymin[1], ymax[1])
    ax.set_ylabel("Weight")
    ax.set_xlabel("Feature")
    ax.grid(False)
    fig.savefig(d + 'Secondeigenvector_with' + typeofdata +'.png')

    fig, ax = plt.subplots()
    ax.bar(feature_names, averagelambda, yerr=std ,ecolor='gray')
    for i, v in enumerate(np.linspace(0, 10, 10)):
        plt.text(i, averagelambda[i] + 0.000002, "{:.2f}".format(averagelambda[i]),
                 ha='center',
                 va="bottom", color='forestgreen', fontsize=16)
    ax.set_ylim(ymin[2], ymax[2])
    ax.set_ylabel("Relevance")
    ax.set_xlabel("Diagonal elements of Λ")
    ax.grid(False)
    fig.savefig(d + 'Relevancematrix_with' + typeofdata + '.png')






# Plot the eigenvalues of the eigenvectors of the relevance matrix.
def ploteigenvalueswithout(eigenvalues, eigenvectors, feature_names, averagelambda, std ,d, typeofdata,ymin, ymax):

    fig, ax = plt.subplots()
    ax.bar(range(0, len(eigenvalues)), eigenvalues)
    if d == "disease":
        for i in range(1):
            plt.text(i, eigenvalues[i],  "{:.2f}".format(eigenvalues[i]),fontsize=14)
    else:
        plt.text(0, eigenvalues[0], "{:.2f}".format(eigenvalues[0]),fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Eigenvalue")
    ax.set_xlabel("Feature")
    ax.axhline(y=eigenvalues[0],color ='red', lw = 2)
    ax.grid(False)
    fig.savefig( d + 'Eigenvalues_without' + typeofdata + '.png')

    fig, ax = plt.subplots()
    ax.bar(feature_names, eigenvectors[0, :])
    for i, v in enumerate(np.linspace(0, 10, 10)):
        plt.text(i, eigenvectors[0, i], "{:.2f}".format(eigenvectors[0, i]), color='k', fontsize=14)
    ax.axhline(0, color='k')
    ax.set_ylim(ymin[0], ymax[0])
    ax.set_ylabel("Weight")
    ax.set_xlabel("Feature")
    ax.grid(False)
    fig.savefig( d + 'Firsteigen_vector_without'+ typeofdata + '.png')

    fig, ax = plt.subplots()
    ax.bar(feature_names, eigenvectors[1, :])
    ax.axhline(0, color='k')
    ax.set_ylim(ymin[1], ymax[1])
    ax.set_ylabel("Weight")
    ax.set_xlabel("Feature")
    ax.grid(False)
    fig.savefig(d + 'Secondeigen_vector_without' + typeofdata + '.png')

    fig, ax = plt.subplots()
    ax.bar(feature_names, averagelambda, yerr=std)
    ax.set_ylim(0, 0.7)
    for i, v in enumerate(np.linspace(0, 10, 10)):
        plt.text(i, averagelambda[i] + 0.000002, "{:.2f}".format(averagelambda[i]),
                 ha='center',
                 va="bottom", color='orangered', fontsize=14)
    ax.set_ylim(ymin[2], ymax[2])
    ax.set_ylabel("Relevance")
    ax.set_xlabel("Diagonal elements of Λ")
    ax.grid(False)
    fig.savefig(d + 'Relevancematrix_without' + typeofdata + '.png')


def ploteigenvaluesingle(eigenvalues, eigenvectors, feature_names, averagelambda, d, typeofdata,ymin, ymax):
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

    # Plot the first two eigenvectors of the relevance matrix, which  is called `omega_hat`.
    fig, ax = plt.subplots()
    ax.bar(feature_names, eigenvectors[0, :])
    for i, v in enumerate(np.linspace(0, 10, 10)):
        plt.text(i, eigenvectors[0, i], "{:.2f}".format(eigenvectors[0, i]), color='k', fontsize=14)
    ax.axhline(0, color='k')
    ax.set_ylim(ymin[0], ymax[0])
    ax.set_ylabel("Weight")
    ax.set_xlabel("Feature")
    ax.grid(False)
    fig.savefig(d + 'Firsteigen_vector_single' + typeofdata + '.png')

    fig, ax = plt.subplots()
    ax.bar(feature_names, eigenvectors[1, :])
    ax.axhline(0, color='k')
    ax.set_ylim(ymin[1], ymax[1])
    ax.set_ylabel("Weight")
    ax.set_xlabel("Feature")
    ax.grid(False)
    fig.savefig(d + 'Secondeigen_vector_single' + typeofdata + '.png')

    fig, ax = plt.subplots()
    ax.bar(feature_names, averagelambda)
    for i, v in enumerate(np.linspace(0, 10, 10)):
        plt.text(i, averagelambda[i], "{:.2f}".format(averagelambda[i]), color='k', fontsize=14)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Relevance")
    ax.set_xlabel("Diagonal elements of Λ")
    ax.grid(False)
    fig.savefig(d + 'Relevance_matrix_single' + typeofdata + '.png')
