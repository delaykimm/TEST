import os
import json
import math
import torch
import numpy
import pandas
import argparse
import scikit_wrappers, llm_wrappers

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from transformers import GPT2Model, GPT2Tokenizer

# import pydevd_pycharm
# pydevd_pycharm.settrace('11.164.204.56', port=31235, stdoutToServer=True, stderrToServer=True)

def load_UCR_dataset(path, dataset, seed =42):
    """
    @param path Path where the UCR dataset is located.
    @param dataset Name of the UCR dataset.

    @return Quadruplet containing the training set, the corresponding training
            labels, the testing set and the corresponding testing labels.
    """
    # 시드 설정
    numpy.random.seed(seed)
    
    train_file = os.path.join(path, dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join(path, dataset, dataset + "_TEST.tsv")
    
    # 데이터 로드 시 크기 제한 추가
    max_samples = 500
    
    # train_df = pandas.read_csv(train_file, sep='\t', header=None)
    # test_df = pandas.read_csv(test_file, sep='\t', header=None)
        
    # train_array = numpy.array(train_df)
    # test_array = numpy.array(test_df)

    # # Move the labels to {0, ..., L-1}
    # labels = numpy.unique(train_array[:, 0])
    # transform = {}
    # for i, l in enumerate(labels):
    #     transform[l] = i

    # train = numpy.expand_dims(train_array[:, 1:], 1).astype(numpy.float64)
    # train_labels = numpy.vectorize(transform.get)(train_array[:, 0])
    # test = numpy.expand_dims(test_array[:, 1:], 1).astype(numpy.float64)
    # test_labels = numpy.vectorize(transform.get)(test_array[:, 0])

    # 데이터 로드
    train = numpy.loadtxt(train_file, delimiter='\t')
    test = numpy.loadtxt(test_file, delimiter='\t')
    
    # 학습 데이터 크기 제한 및 균일 샘플링
    if len(train) > max_samples:
        print(f"Original training set size: {len(train)}")
        
        # 클래스별 데이터 분리
        unique_labels = numpy.unique(train[:, 0])
        n_classes = len(unique_labels)
        
        # 클래스당 샘플 수 계산
        samples_per_class = max_samples // n_classes
        print(f"Samples per class: {samples_per_class}")
        
        # 균일 샘플링
        balanced_indices = []
        for label in unique_labels:
            label_indices = numpy.where(train[:, 0] == label)[0]
            
            if len(label_indices) > samples_per_class:
                # 랜덤 샘플링
                selected_indices = numpy.random.choice(
                    label_indices, 
                    size=samples_per_class, 
                    replace=False,
                    p = None
                )
            else:
                # 모든 샘플 사용 및 부족한 경우 중복 샘플링
                selected_indices = numpy.random.choice(
                    label_indices, 
                    size=samples_per_class, 
                    replace=False,
                    p = None
                )
            
            balanced_indices.extend(selected_indices)
        
        # 최종 데이터 선택
        train = train[balanced_indices]
        
        # 클래스별 분포 출력
        print("\nClass distribution after balancing:")
        unique, counts = numpy.unique(train[:, 0], return_counts=True)
        for label, count in zip(unique, counts):
            print(f"Class {label}: {count} samples")
        
        print(f"Reduced training set size: {len(train)}")
        
    # # 학습 데이터 제한
    # if len(train) > max_samples:
    #     print(f"Original training set size: {len(train)}")
    #     train = train[:max_samples]
    #     print(f"Reduced training set size: {len(train)}")
    
    # 데이터와 레이블 분리
    train_X = train[:, 1:]
    train_y = train[:, 0]
    test_X = test[:, 1:]
    test_y = test[:, 0]

    # Preprocessing
    train_X = numpy.expand_dims(train_X, 1)
    test_X = numpy.expand_dims(test_X, 1)

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        #return train, train_labels, test, test_labels
        return train_X, train_y, test_X, test_y
    # Post-publication note:
    # Using the testing set to normalize might bias the learned network,
    # but with a limited impact on the reported results on few datasets.
    # See the related discussion here: https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/pull/13.
    mean = numpy.nanmean(numpy.concatenate([train, test]))
    var = numpy.nanvar(numpy.concatenate([train, test]))
    train = (train - mean) / math.sqrt(var)
    test = (test - mean) / math.sqrt(var)
    return train, train_labels, test, test_labels

def fit_hyperparameters(file, train, train_labels, cuda, gpu,local_rank,
                        save_memory=False):
    ### hyperparameter 설정 파일을 불러와서 Causal CNN encoder 분류기를 학습. 
    """
    @param file Path of a file containing a set of hyperparemeters.
    @param train Training set.
    @param train_labels Labels for the training set.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    @param save_memory If True, save GPU memory by propagating gradients after
           each loss term, instead of doing it after computing the whole loss.
    """
    #classifier = scikit_wrappers.CausalCNNEncoderClassifier()
    # classifier = llm_wrappers.CausalCNNEncoderLLM(text_prototype=args.text_prototype)

    # Loads a given set of hyperparameters and fits a model with those
    hf = open(os.path.join(file), 'r')
    params = json.load(hf)
    hf.close()
    # Check the number of input channels
    params['in_channels'] = numpy.shape(train)[1]
    params['cuda'] = cuda
    params['gpu'] = gpu
    params['local_rank']= local_rank
    prototype_file = args.prototype_file
    text_prototype = args.text_prototype
    params['reduced_size'] = torch.load(prototype_file, weights_only = True).size(1)
    classifier.set_params(**params)
    return classifier.fit(
        train, train_labels, save_memory=save_memory, verbose=True
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UCR repository datasets'
    )
    parser.add_argument('--dataset', type=str, metavar='D', required=True,
                        help='dataset name')
    parser.add_argument('--path', type=str, metavar='PATH', required=True,
                        help='path where the dataset is located')
    parser.add_argument('--save_path', type=str, metavar='PATH', required=True,
                        help='path where the estimator is/should be saved')
    parser.add_argument('--prototype_file', type=str, metavar='PATH', required=True,
                        help='path where the prototype is/should be loaded')
    parser.add_argument('--text_prototype', action = 'store_true', required = True,
                        help='Whether to use text prototype')
    parser.add_argument('--cuda', action='store_true',
                        help='activate to use CUDA')
    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
                        help='index of GPU used for computations (default: 0)') # use sigle gpu
    parser.add_argument("--local_rank", type=int, default=-1, help='DDP parameter (default: -1)') # DDP
    parser.add_argument('--hyper', type=str, metavar='FILE', required=True,
                        help='path of the file of hyperparameters to use; ' +
                             'for training; must be a JSON file')
    parser.add_argument('--load', action='store_true', default=False,
                        help='activate to load the estimator instead of ' +
                             'training it')
    parser.add_argument('--fit_classifier', action='store_true', default=False,
                        help='if not supervised, activate to load the ' +
                             'model and retrain the classifier')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    if args.cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        args.cuda = False

    train, train_labels, test, test_labels = load_UCR_dataset(
        args.path, args.dataset
    )
    class_num = len(numpy.unique(train_labels))
    
    classifier = llm_wrappers.CausalCNNEncoderLLM(text_prototype=args.text_prototype)
    
    if not args.load and not args.fit_classifier: # 모델을 새로 학습할 경우
        device = torch.device('cuda' if args.cuda else 'cpu')
     # 하이퍼파라미터를 조정하면서 학습을 수행. fit_hyperparameters 함수는 주어진 하이퍼파라미터로 모델을 학습시키고, 최적의 hyperparameter을 찾음.
        classifier = fit_hyperparameters( 
            args.hyper, train, train_labels, args.cuda, args.gpu, args.local_rank
        )
    else: # 저장된 모델을 로드. 
        #classifier = scikit_wrappers.CausalCNNEncoderClassifier()
        classifier = llm_wrappers.CausalCNNEncoderLLM(text_prototype = args.text_prototype)
        hf = open(
            os.path.join(
                args.save_path, args.dataset + '_hyperparameters.json'
            ), 'r'
        )
        hp_dict = json.load(hf) # 저장된 하이퍼파라미터를 JSON 파일에서 불러옴.
        hf.close()
        hp_dict['cuda'] = args.cuda
        hp_dict['gpu'] = args.gpu
        hp_dict['local_rank'] = args.local_rank
        prototype_file = args.prototype_file
        hp_dict['reduced_size'] = torch.load(prototype_file).size(1)
        classifier.set_params(**hp_dict) # 불러온 hyperparameter을 모델에 설정. 
        classifier.load(os.path.join(args.save_path, args.dataset)) # 저장된 모델 가중치와 상태를 불러옴. 

    if not args.load:             # 분류기 재학습
        if args.fit_classifier:   # 분류기만 다시 학습
            classifier.fit_classifier(classifier.encode(train), train_labels)
        classifier.save(   # 모델 저장
            os.path.join(args.save_path, args.dataset)
        )
        with open(
            os.path.join(    
                args.save_path, args.dataset + '_hyperparameters.json'
            ), 'w'
        ) as fp:    # 하이퍼파라미터 저장
            json.dump(classifier.get_params(), fp)

    #print("Test accuracy: " + str(classifier.score(test, test_labels)))
