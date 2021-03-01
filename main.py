from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES

from utils.utils import read_all_datasets
from utils.utils import transform_labels
from utils.utils import create_directory
from utils.utils import run_length_xps
from utils.utils import generate_results_csv
import utils
import numpy as np
import sys
import sklearn
import os 
from sklearn import preprocessing
import pandas as pd
import random
from sklearn.model_selection import train_test_split

root_dir= '/content/drive/MyDrive/Cadeiras/Mestrado/DEEP'
#root_dir = '/dataset'
xps = ['use_bottleneck', 'use_residual', 'nb_filters', 'depth',
       'kernel_size', 'batch_size']
ARCHIVES_FOLDER = '/content/drive/MyDrive/Cadeiras/Mestrado/DEEP/archives/GAS'
GAS_FOLDERS = [ 'Toluene_200', 'Methanol_200', 'Methane_1000', 'Ethylene_500', 'CO_4000', 'CO_1000', 'Butanol_100', 
                'Benzene_200', 'Ammonia_10000', 'Acetone_2500', 'Acetaldehyde_500']
GAS_CLASS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
BOARD_POSITIONS = ['L1', 'L2', 'L2', 'L4', 'L5', 'L6']
FINAL_COLUMN_TYPE = { 'class': 'int32', 'mcf1_setpoint': 'float16', 'mcf2_setpoint': 'float16', 'mcf3_setpoint': 'float16', 'mcf1_read': 'float16', 'mcf2_read': 'float16', 'mcf3_read': 'float16', 't': 'float16', 'rh': 'float16', 'b1': 'float16', 'board11': 'float16', 'board12': 'float16', 'board13': 'float16', 'board14': 'float16', 'board15': 'float16', 'board16': 'float16', 'board17': 'float16', 'board18': 'float16', 'b2': 'float16', 'board21': 'float16', 'board22': 'float16', 'board23': 'float16', 'board24': 'float16', 'board25': 'float16', 'board26': 'float16', 'board27': 'float16', 'board28': 'float16', 'b3': 'float16', 'board31': 'float16', 'board32': 'float16', 'board33': 'float16', 'board34': 'float16', 'board35': 'float16', 'board36': 'float16', 'board37': 'float16', 'board38': 'float16', 'b4': 'float16', 
'board41': 'float16', 'board42': 'float16', 'board43': 'float16', 'board44': 'float16', 'board45': 'float16', 'board46': 'float16', 'board47': 'float16', 'board48': 'float16', 'b5': 'float16', 'board51': 'float16', 'board52': 'float16', 'board53': 'float16', 'board54': 'float16', 'board55': 'float16', 'board56': 'float16', 'board57': 'float16', 'board58': 'float16', 'b6': 'float16', 
'board61': 'float16', 'board62': 'float16', 'board63': 'float16', 'board64': 'float16', 'board65': 'float16', 'board66': 'float16', 'board67': 'float16', 'board68': 'float16', 'b7': 'float16', 'board71': 'float16', 'board72': 'float16', 'board73': 'float16', 'board74': 'float16', 'board75': 'float16', 'board76': 'float16', 'board77': 'float16', 'board78': 'float16', 'b8': 'float16', 'board81': 'float16', 'board82': 'float16', 'board83': 'float16', 'board84': 'float16', 'board85': 'float16', 'board86': 'float16', 'board87': 'float16', 'board88': 'float16', 'b9': 'float16', 
'board91': 'float16', 'board92': 'float16', 'board93': 'float16', 'board94': 'float16', 'board95': 'float16', 'board96': 'float16', 'board97': 'float16', 'board98': 'float16'}


def readnpy_return_train_test():
  filename = ARCHIVES_FOLDER + '/cleaned_normalized_256_data.npy'
  data = np.load(filename)
  Y = data[:, 0]
  X = data[:, 1:]
  x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y)

  nb_classes = 11

  y_true = y_test.astype(np.int64)
  y_true_train = y_train.astype(np.int64)
  # transform the labels from integers to one hot vectors
  enc = sklearn.preprocessing.OneHotEncoder()
  enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
  y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
  y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

  if len(x_train.shape) == 2:  # if univariate
    # add a dimension to make it multivariate with one dimension
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

  return x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train

def prepare_data():
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # make the min to zero of labels
    y_train, y_test = transform_labels(y_train, y_test)

    # save orignal y because later we will use binary
    y_true = y_test.astype(np.int64)
    y_true_train = y_train.astype(np.int64)
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc


def fit_classifier():
    input_shape = x_train.shape[1:]

    classifier = create_classifier(classifier_name, input_shape, nb_classes,
                                   output_directory)

    classifier.fit(x_train, y_train, x_test, y_test, y_true)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory,
                      verbose=False, build=True):
    if classifier_name == 'nne':
        from classifiers import nne
        return nne.Classifier_NNE(output_directory, input_shape,
                                  nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose,
                                              build=build)


def get_xp_val(xp):
    if xp == 'batch_size':
        xp_arr = [16, 32, 128]
    elif xp == 'use_bottleneck':
        xp_arr = [False]
    elif xp == 'use_residual':
        xp_arr = [False]
    elif xp == 'nb_filters':
        xp_arr = [16, 64]
    elif xp == 'depth':
        xp_arr = [3, 9]
    elif xp == 'kernel_size':
        xp_arr = [8, 64]
    else:
        raise Exception('wrong argument')
    return xp_arr


############################################### main
if sys.argv[1] == 'InceptionTime':
    # run nb_iter_ iterations of Inception on the whole TSC archive
    classifier_name = 'inception'
    archive_name = ARCHIVE_NAMES[0]
    nb_iter_ = 5
    #datasets_dict = read_all_datasets(root_dir, archive_name)

    x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train = readnpy_return_train_test()

    for iter in range(nb_iter_):
        print('\t\titer', iter)

        trr = ''
        if iter != 0:
            trr = '_itr_' + str(iter)

        tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + trr + '/'

        for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
            print('\t\t\tdataset_name: ', dataset_name)

            #x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data()

            output_directory = tmp_output_directory + dataset_name + '/'

            temp_output_directory = create_directory(output_directory)

            if temp_output_directory is None:
                print('Already_done', tmp_output_directory, dataset_name)
                continue

            fit_classifier()

            print('\t\t\t\tDONE')

            # the creation of this directory means
            create_directory(output_directory + '/DONE')

    # run the ensembling of these iterations of Inception
    classifier_name = 'nne'
    #datasets_dict = read_all_datasets(root_dir, archive_name)
    tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/'

    for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
        print('\t\t\tdataset_name: ', dataset_name)

        #x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data()

        output_directory = tmp_output_directory + dataset_name + '/'

        fit_classifier()

        print('\t\t\t\tDONE')

elif sys.argv[1] == 'InceptionTime_xp':
    # this part is for running inception with the different hyperparameters
    # listed in the paper, on the whole TSC archive
    archive_name = 'TSC'
    classifier_name = 'inception'
    max_iterations = 5

    datasets_dict = read_all_datasets(root_dir, archive_name)

    for xp in xps:

        xp_arr = get_xp_val(xp)

        print('xp', xp)

        for xp_val in xp_arr:
            print('\txp_val', xp_val)

            kwargs = {xp: xp_val}

            for iter in range(max_iterations):

                trr = ''
                if iter != 0:
                    trr = '_itr_' + str(iter)
                print('\t\titer', iter)

                for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:

                    output_directory = root_dir + '/results/' + classifier_name + '/' + '/' + xp + '/' + '/' + str(
                        xp_val) + '/' + archive_name + trr + '/' + dataset_name + '/'

                    print('\t\t\tdataset_name', dataset_name)
                    x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data()

                    # check if data is too big for this gpu
                    size_data = x_train.shape[0] * x_train.shape[1]

                    temp_output_directory = create_directory(output_directory)

                    if temp_output_directory is None:
                        print('\t\t\t\t', 'Already_done')
                        continue

                    input_shape = x_train.shape[1:]

                    from classifiers import inception

                    classifier = inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes,
                                                                verbose=False, build=True, **kwargs)

                    classifier.fit(x_train, y_train, x_test, y_test, y_true)

                    # the creation of this directory means
                    create_directory(output_directory + '/DONE')

                    print('\t\t\t\t', 'DONE')

    # we now need to ensemble each iteration of inception (aka InceptionTime)
    archive_name = ARCHIVE_NAMES[0]
    classifier_name = 'nne'

    datasets_dict = read_all_datasets(root_dir, archive_name)

    tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/'

    for xp in xps:
        xp_arr = get_xp_val(xp)
        for xp_val in xp_arr:

            clf_name = 'inception/' + xp + '/' + str(xp_val)

            for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
                x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data()

                output_directory = tmp_output_directory + dataset_name + '/'

                from classifiers import nne

                classifier = nne.Classifier_NNE(output_directory, x_train.shape[1:],
                                                nb_classes, clf_name=clf_name)

                classifier.fit(x_train, y_train, x_test, y_test, y_true)

elif sys.argv[1] == 'run_length_xps':
    # this is to generate the archive for the length experiments
    run_length_xps(root_dir)

elif sys.argv[1] == 'generate_results_csv':
    clfs = []
    itr = '-0-1-2-3-4-'
    inceptionTime = 'nne/inception'
    # add InceptionTime: an ensemble of 5 Inception networks
    clfs.append(inceptionTime + itr)
    # add InceptionTime for each hyperparameter study
    for xp in xps:
        xp_arr = get_xp_val(xp)
        for xp_val in xp_arr:
            clfs.append(inceptionTime + '/' + xp + '/' + str(xp_val) + itr)
    df = generate_results_csv('results.csv', root_dir, clfs)
    print(df)
