import math
import numpy
import torch
import torch.nn.parallel
import sklearn
import sklearn.svm
import joblib
import sklearn.model_selection
import utils
import losses
import networks
import os
import random
import torch.nn as nn
from networks.llm_gpt2 import GPT2FeatureExtractor
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import make_interp_spline

def set_seed(seed=42):
    """모든 난수 생성기의 시드를 설정합니다."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class Data:
    # def __init__(self, max_seq_len, feature_df):
    #     self.max_seq_len = max_seq_len  # 시퀀스의 최대 길이
    #     self.feature_df = feature_df    # 입력 데이터의 특징 벡터 (특성 차원)
    def __init__(self, max_seq_len=None, feature_df=None):
        self.max_seq_len = max_seq_len if max_seq_len is not None else 128  # 기본값 설정
        self.feature_df = feature_df if feature_df is not None else torch.zeros((1, 768))  # GPT2 기본 차원

class TimeSeriesEncoderLLM(sklearn.base.BaseEstimator,
                                  sklearn.base.ClassifierMixin):
    """
    Class to wrap an encoder of time series and LLM's self_attention blocks
    and a classifier on top of computed representations.

    All inheriting classes should implement the get_params and set_params
    methods.

    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of randomly chosen intervals to select the
           final negative sample in the loss.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    @param batch_size Batch size used during the training of the encoder.
    @param nb_steps Number of optimization steps to perform for the training of
           the encoder.
    @param lr learning rate of the Adam optimizer used to train the encoder.
    @param penalty Penalty term for the SVM classifier. If None and if the
           number of samples is high enough, performs a hyperparameter search
           to find a suitable constant.
    @param early_stopping Enables, if not None, early stopping heuristic
           for the training of the representations, based on the final
           score. Representations are still learned unsupervisedly in this
           case. If the number of samples per class is no more than 10,
           disables this heuristic. If not None, accepts an integer
           representing the patience of the early stopping strategy.
    @param encoder Encoder PyTorch module.
    @param params Dictionaries of the parameters of the encoder.
    @param in_channels Number of input channels of the time series.
    @param cuda Transfers, if True, all computations to the GPU.
    @param gpu GPU index to use, if CUDA is enabled.
    @param text_prototype Selected text prototypes.
    @param llm_model_dir Download llm mode dir, select embedding to initialize prompts.
    """
    def __init__(self, compared_length, nb_random_samples, negative_penalty,
                 batch_size, nb_steps, lr, penalty, early_stopping,
                 encoder, params, in_channels, out_channels,text_prototype,cuda=False,
                 gpu=0,local_rank=-1, seed = 42):
        # 시드 설정 추가
        self.seed = seed
        set_seed(self.seed)
        
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.local_rank=local_rank
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.penalty = penalty
        self.early_stopping = early_stopping
        self.encoder = encoder
        self.params = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_prototype = True # text_prototype 
        
        print(self.text_prototype)
        # text_prototype이 True일 때만 프로토타입 임베딩 생성
        self.prompt_matrix = None
        if self.text_prototype:
            self.prompt_matrix = losses.text_prototype.select_prototype(
                model_name='gpt2',
                prototype_dir = "./gpt2_prototype", #None,
                provide=False, 
                number_of_prototype=3
            )[0]
            print(self.prompt_matrix.shape)
            # 프로토타입이 변경될 때마다 새로운 값으로 업데이트되도록 수정
            self.prompt_matrix = torch.nn.Parameter(self.prompt_matrix, requires_grad=True)
            
            if self.cuda:
                if self.local_rank != -1:
                    self.prompt_matrix = self.prompt_matrix.cuda(self.local_rank)
                else:
                    self.prompt_matrix = self.prompt_matrix.cuda(self.gpu)
            
        # Data 객체는 나중에 실제 특징이 추출된 후 업데이트
        self.data = Data(
            max_seq_len=None,  # encode_window에서 실제 특징이 추출된 후 업데이트
            feature_df=torch.zeros((1, out_channels))   # encode_window의 결과로 업데이트
        )
        # config와 data 정의
        config = {
            'patch_size': 8,  # 입력 데이터를 처리할 패치 크기 (window 크기)
            'stride': 8,       # 슬라이딩 윈도우의 보폭 (stride)
            'd_model': 768,    # 모델의 차원, GPT-2의 기본 임베딩 차원 (GPT-2 모델의 기본 차원은 768)
            'dropout': 0.1,     # 드롭아웃 비율
             # num_classes를 초기화 시점에는 None으로 설정
            'num_classes': None  # fit 메서드에서 실제 클래스 수로 업데이트될 예정           
        }
        
        # Loss Function 초기화 시 text_prototype 전달
        self.llm = GPT2FeatureExtractor(config, self.data)
        self.loss = losses.contrastive_loss.ContrastiveLoss(
            compared_length, nb_random_samples, negative_penalty,
            text_prototype = self.text_prototype#, text_prototype_embeddings = self.prompt_matrix
        )
        self.loss_varying = losses.contrastive_loss.ContrastiveLossVaryingLength(
            compared_length, nb_random_samples, negative_penalty,
            text_prototype = self.text_prototype
        )
        self.classifier = sklearn.svm.SVC()
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        
        # 여기에 GPT-2 임베딩 초기화 코드 추가
        self.initialize_embeddings() 

    def semantic_matching(self, embeddings, X, labels, title="Matching TS Embedding to Words", save_path='./visualization_semantic/semantic_matching.png'):
        """
        시계열 데이터의 구간별 임베딩을 시각화하고 가장 가까운 단어와 매칭하여 표시합니다.
        
        @param embeddings: 구간별 임베딩 벡터 (features 또는 features_from_llm)
        @param X: 원본 시계열 데이터
        @param labels: 클래스 라벨
        @param title: 그래프 제목
        @param save_path: 저장 경로
        """
        # 시드 재설정
        set_seed(self.seed)
        
        # 텐서를 numpy 배열로 변환
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # 클래스당 하나의 대표 시계열 선택
        unique_labels = numpy.unique(labels)
        #print("unique_labels: ", unique_labels)
        selected_embeddings = []
        selected_timeseries = []
        selected_labels = []
        
        for label in unique_labels:
            label_indices = numpy.where(labels == label)[0]
            #print("label_indices : ", label_indices)
            selected_idx = label_indices[0]  # 각 클래스의 첫 번째 샘플 선택
            selected_embeddings.append(embeddings[selected_idx])
            selected_timeseries.append(X[selected_idx])
            selected_labels.append(label)
        
        selected_embeddings = numpy.array(selected_embeddings)
        selected_timeseries = numpy.array(selected_timeseries)
        
        print("Embeddings shape before processing:", embeddings.shape)
        print("Selected embeddings shape: ", selected_embeddings[0].shape, selected_embeddings.shape)
        print("Selected timeseries shape: ", selected_timeseries[0].shape, selected_timeseries.shape)
        
        selected_timeseries = selected_timeseries.reshape(selected_timeseries.shape[0], -1)
        
        # 시각화를 위한 figure 설정
        num_rows = selected_embeddings.shape[0]  # 동적 행 개수 설정
        print(num_rows)
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(num_rows+1, 1, height_ratios=[7] + [1] * num_rows, figure=fig)
        fig.suptitle(title, fontsize=16)
        
        matched_words_all = []  # 전체 매칭 단어 저장
        
        # 각 시계열의 구간별 임베딩에 대해 처리
        for idx, (embedding, timeseries, label) in enumerate(zip(selected_embeddings, selected_timeseries, selected_labels)):
            
            print("Current embedding shape:", embedding.shape)
            
            # 데이터 reshape
            if embedding.ndim == 1:
                embedding = embedding.reshape(-1, 768)
            elif embedding.ndim == 3:
                embedding = embedding.reshape(embedding.shape[0], -1)
                
            # Perform PCA to reduce embeddings to 3D
            tsne = TSNE(n_components=3, perplexity=5, n_iter=300, random_state = self.seed)
            reduced_embedding = tsne.fit_transform(embedding)
            # pca = PCA(n_components=3, random_state=self.seed)
            # reduced_embedding = pca.fit_transform(embedding)
            
            print("After embedding shape:", embedding.shape)
            
            # # Initialize plot
            # fig = plt.figure(figsize=(12, 14))
            # gs = GridSpec(3, 1, height_ratios=[2, 1, 1], figure=fig)
            # fig.suptitle(title, fontsize=16)
            
            # 3D plot of embeddings
            ax_main = fig.add_subplot(gs[0], projection='3d')
            ax_main.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1], reduced_embedding[:, 2], s=5, color='gray', alpha=0.5)
            
            # Match words to each embedding segment and plot
            matched_words = []
            path_points = []

            for i in range(embedding.shape[0]):
                similarities = cosine_similarity(embedding[i].reshape(1, -1), self.vocab_embeddings)[0]
                closest_word = self.vocab_words[numpy.argmax(similarities)]
                matched_words.append(closest_word)
                path_points.append(reduced_embedding[i])
                
                # # Display each word at its 3D position
                # ax_main.text(reduced_embedding[i, 0], reduced_embedding[i, 1], reduced_embedding[i, 2], 
                #             closest_word, fontsize=8, color="black")
            
            matched_words_all.append(matched_words) # 전체 매칭 단어에 추가
            
            # Draw path connecting words in the order they appear
            path_points = numpy.array(path_points)
            
            # 부드러운 곡선 경로를 위해 스플라인 보간
            t = numpy.linspace(0, 1, len(path_points))
            spl = make_interp_spline(t, path_points, k=3)  # 3차 스플라인 보간
            t_fine = numpy.linspace(0, 1, 100)
            smooth_path = spl(t_fine)

            # 경로 그리기 (부드러운 곡선)
            ax_main.plot(smooth_path[:, 0], smooth_path[:, 1], smooth_path[:, 2], color="red", alpha=0.7)
            
            # 시작점과 끝점에만 표시
            start = path_points[0]
            end = path_points[-1]
            ax_main.scatter(*start, color='black', s=100, marker="<")      # 시작 지점에 화살표
            ax_main.scatter(*end, color='black', s=100, label="End")       # 끝 지점에 동그라미
            
            ax_main.set_title("3D Embeddings with Matched Words")
            ax_main.grid(True) # 3D 그리드 활성화
            ax_main.axis('on')
                
            # 하단에 원시 시계열 데이터와 매칭된 단어 시각화
            ax_timeseries = fig.add_subplot(gs[1 + idx])
            # x축을 0부터 시계열 데이터 길이까지 생성하여, 시작점에서 시각화가 시작되도록 함
            x_values = range(len(timeseries))
            ax_timeseries.plot(x_values, timeseries, color="black")
            
            ax_timeseries.set_xticks([])
            ax_timeseries.set_yticks([])
            
            # 각 구간에 해당하는 단어 표시
            segment_length = len(timeseries) // len(matched_words)
            print("segment_length: ", segment_length)
            print("timeseries_length: ", len(timeseries))
            print("matched_words length: ", len(matched_words))
            for j, word in enumerate(matched_words):
                # 각 구간의 중앙에 단어를 표시
                start_idx = j * segment_length
                end_idx = start_idx + segment_length
                segment_center = (start_idx + end_idx) // 2
                
                # 단어를 박스 상단에 표시
                ax_timeseries.text(segment_center, numpy.max(timeseries) + 0.1 * numpy.abs(numpy.max(timeseries)), word, 
                                fontsize=8, ha="center", color="blue" if j % 2 == 0 else "red")  # 색상 교차

                # 박스 경계선 그리기
                ax_timeseries.axvline(x=start_idx, color="black", linestyle="-")  # 박스 시작선
                
                # x축 제한 설정 (시계열 데이터를 각 박스에 꽉 차도록 표시)
            ax_timeseries.set_xlim(0, len(timeseries))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 축 레이블 설정
        ax_main.set_xlabel("X")
        ax_main.set_ylabel("Y")
        ax_main.set_zlabel("Z")

        # Save the visualization
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        plt.close()
        
        return matched_words_all

    def plot_embeddings_2d(self, embeddings, title="2D Embeddings Visualization", labels=None, method='tsne', save_path='../visualization/embeddings_2d.png'):
        """
        주어진 임베딩 벡터를 2D 시각화하는 함수.  (classification 기준)
        
        @param embeddings: 고차원의 임베딩 벡터 (features 또는 features_from_llm)
        @param title: 그래프의 제목
        @param labels: 클래스 라벨 (있을 경우 각 클래스별로 색상 지정)
        @param method: 차원 축소 방법 ('tsne' 또는 'pca')
        """
        
        # 3차원 배열일 경우, 2차원으로 변환
        if embeddings.ndim == 3:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)  # (batch_size, sequence_length * feature_dim)
            
        # GPU에 있는 텐서일 경우 CPU로 이동한 후 numpy로 변환
        if isinstance(embeddings, torch.Tensor):
            if embeddings.is_cuda:
                embeddings = embeddings.cpu().numpy()
            else:
                embeddings = embeddings.numpy()
        
            # 데이터 스케일링 (t-SNE 및 PCA 안정성을 위해)
        try:
            scaler = StandardScaler()
            embeddings = scaler.fit_transform(embeddings)
        except Exception as e:
            print(f"데이터 스케일링 중 오류 발생: {e}")
            
        embeddings += numpy.random.normal(0, 1e-5, size=embeddings.shape)
            
        print("데이터 분산:", numpy.var(embeddings, axis=0))
        
        if method == 'tsne':
                    # t-SNE로 2D 차원 축소
            try:
                tsne = TSNE(n_components=2, perplexity=min(30, embeddings.shape[0] // 3), n_iter=1000, random_state=42)
                reduced_embeddings = tsne.fit_transform(embeddings)
                print("applying tSNE")
            except Exception as e:
                print(f"t-SNE 처리 중 오류 발생: {e}")
                return None
        elif method == 'pca':
            # PCA로 2D 차원 축소
            pca = PCA(n_components=2, random_state=42)
            reduced_embeddings = pca.fit_transform(embeddings)
        else:
            raise ValueError("Unknown method: Use 'tsne' or 'pca'")
        
        # 2D 시각화
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)  # 3D가 아니므로 projection='3d' 제거
        
        if labels is not None:
            # 클래스별로 색을 다르게 표시
            for label in numpy.unique(labels):
                indices = numpy.where(labels == label)
                ax.scatter(reduced_embeddings[indices, 0], 
                        reduced_embeddings[indices, 1], 
                        label=f'Class {label}', alpha=0.6)
            ax.legend()
        else:
            # 클래스 라벨이 없을 경우, 모든 점을 동일하게 표시
            ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.6)
        
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.show()
        
        # 그래프 저장
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to: {save_path}")
        
        plt.close()
        return reduced_embeddings
    
    def save_encoder(self, prefix_file): # 학습된 encoder만 저장
        """
        Saves the encoder and the SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        if self.local_rank==0:
            torch.save(
                self.encoder.module.state_dict(),
                prefix_file + '_' + self.architecture + '_encoder.pth'
            )
        else:
            torch.save(
                self.encoder.state_dict(),
                prefix_file + '_' + self.architecture + '_encoder.pth'
            )

    def save(self, prefix_file): # encoder, classifier 모두 저장
        """
        Saves the encoder and the SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_classifier.pkl' and
               '$(prefix_file)_$(architecture)_encoder.pth').
        """
        self.save_encoder(prefix_file)
        joblib.dump(
            self.classifier,
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def load_encoder(self, prefix_file):
        """
        Loads an encoder.

        @param prefix_file Path and prefix of the file where the model should
               be loaded (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        if self.cuda:
            # load model on each gpu
            if self.local_rank!=-1:
                print("Local rank "+str(self.local_rank) +" is loaded")
                self.encoder.load_state_dict(torch.load(
                    prefix_file + '_' + self.architecture + '_encoder.pth',
                    map_location=lambda storage, loc: storage.cuda(self.local_rank)
                ))
            else:
                print("Load model on GPU " +str(self.gpu))
                self.encoder.load_state_dict(torch.load(
                        prefix_file + '_' + self.architecture + '_encoder.pth',
                        map_location=lambda storage, loc: storage.cuda(self.gpu)
                    ))
        else:
            print("Load model on CPU")
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage
            ))

    def load(self, prefix_file):
        """
        Loads an encoder and an SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be loaded (at '$(prefix_file)_$(architecture)_classifier.pkl'
               and '$(prefix_file)_$(architecture)_encoder.pth').
        """
        self.load_encoder(prefix_file)
        self.classifier = joblib.load(
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def fit_classifier(self, features, y):
        """
        Trains the classifier using precomputed features. Uses an SVM
        classifier with RBF kernel.

        @param features Computed features of the training set.
        @param y Training labels.
        """
        # 시드 재설정
        set_seed(self.seed)
        
        # GPU 텐서를 CPU로 이동하고 numpy 배열로 변환
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
            
        nb_classes = numpy.shape(numpy.unique(y, return_counts=True)[1])[0]
        train_size = numpy.shape(features)[0]
        self.classifier = sklearn.svm.SVC( # parameter optimization(모델 내부의 weight...)
            C=1 / self.penalty
            if self.penalty is not None and self.penalty > 0
            else 1.0,
            gamma='scale', 
            random_state = self.seed # 시드 설정
        )

        if train_size // nb_classes < 5 or train_size < 50 or self.penalty is not None:
            return self.classifier.fit(features, y)
        else: # hyperparameter tuning(모델 외부의 설정 값 - 분류기 학습 전 최적의 모델 설정 학습)
            grid_search = sklearn.model_selection.GridSearchCV(
                self.classifier, {
                    'C': [
                        0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000
                        # numpy.inf
                    ],
                    'kernel': ['rbf'],
                    'degree': [3],
                    'gamma': ['scale'],
                    'coef0': [0],
                    'shrinking': [True],
                    'probability': [False],
                    'tol': [0.001],
                    'cache_size': [200],
                    'class_weight': [None],
                    'verbose': [False],
                    'max_iter': [10000000],
                    'decision_function_shape': ['ovr'],
                    'random_state': [self.seed]
                },
                cv=5, n_jobs=5, #idd = False
            )
            if train_size <= 10000:
                grid_search.fit(features, y)
            else:
                # If the training set is too large, subsample 10000 train
                # examples
                split = sklearn.model_selection.train_test_split(
                    features, y,
                    train_size=10000, random_state=self.seed, stratify=y
                )
                grid_search.fit(split[0], split[2])
            self.classifier = grid_search.best_estimator_
            return self.classifier

    def fit_encoder(self, X, y=None, save_memory=False, verbose=False):
        """
        Trains the encoder unsupervisedly using the given training data.

        @param X Training set.
        @param y Training labels, used only for early stopping, if enabled. If
               None, disables early stopping in the method.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda()

        if y is not None:
            nb_classes = numpy.shape(numpy.unique(y, return_counts=True)[1])[0]
            train_size = numpy.shape(X)[0]
            ratio = train_size // nb_classes

        train_torch_dataset = utils.Dataset(X)

        train_sampler=None
        if self.local_rank != -1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_torch_dataset, seed = self.seed)
            train_generator = torch.utils.data.DataLoader(train_torch_dataset, batch_size=self.batch_size, shuffle=False,
                                                   pin_memory=True,
                                                   sampler=train_sampler)
        else:
            train_generator=torch.utils.data.DataLoader(train_torch_dataset,batch_size=self.batch_size,shuffle=False)

        max_score = 0
        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs
        count = 0  # Count of number of epochs without improvement
        # Will be true if, by enabling epoch_selection, a model was selected
        # using cross-validation
        found_best = False
        
        features = self.encode(X, batch_size = 50)
        self.classifier = self.fit_classifier(features, y)
        # Cross validation score
        score = numpy.mean(sklearn.model_selection.cross_val_score(
            self.classifier, features, y=y, cv=5, n_jobs=5
        ))
        print("encoder score before fit_encoder:" , score)

        ######## main training loop ############
        # Encoder training
        while i < self.nb_steps:
            if verbose and (self.local_rank==0 or self.local_rank==-1):
                print('Epoch: ', epochs + 1)
            if self.local_rank != -1:
                train_sampler.set_epoch(i)
                
            # 학습 시작 전 파라미터 상태 저장
            initial_params = {name: param.clone().detach() for name, param in self.encoder.named_parameters()}
            
            for batch in train_generator:
                if self.cuda:
                    if self.local_rank != -1:
                        batch = batch.cuda(self.local_rank)
                    else:
                        batch=batch.cuda(self.gpu)
                
                before_step = {name: param.clone().detach() for name, param in self.encoder.named_parameters()}
                        
                self.optimizer.zero_grad()
                
                 # 여기서 loss 계산 부분 수정
                if self.text_prototype and self.prompt_matrix is not None:
                    current_prototype = self.prompt_matrix.clone()
                    loss = self.loss(
                        batch=batch,
                        encoder=self.encoder,
                        train=train,
                        text_prototype_embeddings=current_prototype,  # prompt_matrix 전달
                        save_memory=save_memory
                    )
                else:
                    loss = self.loss(
                        batch=batch,
                        encoder=self.encoder,
                        train=train,
                        save_memory=save_memory
                    )
                print("debug!") 
                # torch.autograd.set_detect_anomaly(True)  # 디버깅 모드 활성화
                loss.backward()
                
                # gradient 확인
                for name, param in self.encoder.named_parameters():
                    if param.grad is not None:
                        print(f"Gradient norm for {name}: {param.grad.norm().item()}")
               
                self.optimizer.step()
                i += 1
                print(i)
                # 배치 학습 후 파라미터 저장
                after_step = {name: param.clone().detach() for name, param in self.encoder.named_parameters()}

                for name in before_step:
                    param_diff = (after_step[name] - before_step[name]).norm().item()
                    if param_diff > 0:
                        print(f"Parameter {name} changed by {param_diff}")
                
                if i >= self.nb_steps:
                    print("Finished")
                    break
            epochs += 1
            # Early stopping strategy
            # 조기 종료가 활성화된 경우, 매번 에포크가 끝날 때마다 교차 검증을 통해 성능을 평가합니다.
	          # 최고 성능을 기록한 에포크에서 모델 상태를 저장하고, 조기 종료 조건에 따라 학습을 종료합니다.
            # Early stopping strategy
            if self.early_stopping is not None and y is not None and (
                ratio >= 5 and train_size >= 50
            ):
                print("early_stopping is not None")
                
                # Computes the best regularization parameters
                features = self.encode(X, batch_size = 50)
                self.classifier = self.fit_classifier(features, y)
                # Cross validation score
                score = numpy.mean(sklearn.model_selection.cross_val_score(
                    self.classifier, features, y=y, cv=5, n_jobs=5
                ))
                print("encoder score during fit_encoder:" , score)
                count += 1
                # If the model is better than the previous one, update
                if score > max_score:
                    count = 0
                    found_best = True
                    max_score = score
                    best_encoder = type(self.encoder)(**self.params)
                    best_encoder.double()
                    
                    if self.cuda:
                        if self.local_rank!=-1:
                            best_encoder.cuda(self.local_rank)
                        else:
                            best_encoder.cuda(self.gpu)
                    best_encoder.load_state_dict(self.encoder.state_dict())
                    
            if count == self.early_stopping:
                # Computes the best regularization parameters
                features = self.encode(X, batch_size = 50)
                self.classifier = self.fit_classifier(features, y)
                # Cross validation score
                score = numpy.mean(sklearn.model_selection.cross_val_score(
                    self.classifier, features, y=y, cv=5, n_jobs=5
                ))
                print("encoder score during fit_encoder:" , score)
                break

        # If a better model was found, use it
        if found_best:
            self.encoder = best_encoder

        return self.encoder
    
    def process_features_window(self, features, batch_size=256):      
        # 시드 재설정
        set_seed(self.seed)
        
        # numpy array를 torch.tensor로 변환
        if isinstance(features, numpy.ndarray):
            features = torch.from_numpy(features)
        
        # (B, C, W) -> (B, W, C) 변환하여 GPT2FeatureExtractor 입력 형태에 맞춤
        features = features.transpose(1, 2)  # (B, W, C)
        
        # DataLoader 생성
        dataset = torch.utils.data.TensorDataset(features)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False, 
            worker_init_fn=lambda worker_id: set_seed(self.seed + worker_id)
        )
        
        # 배치 단위로 처리
        all_outputs = []
        for batch in dataloader:
            batch_features = batch[0]  # (B, W, C)
            
            if self.cuda:
                batch_features = batch_features.cuda(self.gpu)
                
            # GPT2FeatureExtractor로 처리
            with torch.no_grad():
                batch_outputs = self.llm(batch_features)
                
            if self.cuda:
                batch_outputs = batch_outputs.cpu()
                
            all_outputs.append(batch_outputs)
        
        # 모든 배치의 결과를 합침
        features_from_llm = torch.cat(all_outputs, dim=0)
        print(features_from_llm.shape)
        return features_from_llm

    def fit(self, X, y, save_memory=False, verbose=False):
        """
        Trains sequentially the encoder unsupervisedly and then the classifier
        using the given labels over the learned features.

        @param X Training set.
        @param y Training labels.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """
        set_seed(self.seed)
        
        # 실제 클래스 수를 계산하고 config 업데이트
        num_classes = len(numpy.unique(y))
        print("num_classes: ", num_classes)
        self.llm.num_classes = num_classes  # GPT2FeatureExtractor의 num_classes 직접 업데이트
        
        # Fitting encoder
        self.encoder = self.fit_encoder(
            X, y=y, save_memory=save_memory, verbose=verbose
        )

        # SVM classifier training
        ###	1.	인코더 학습: fit_encoder 메서드를 사용해 비지도 학습 방식으로 시계열 데이터를 인코딩하는 모델(인코더)을 학습합니다.
	    ### 2.	특징 추출 및 LLM 변환: 학습된 인코더를 사용해 입력 데이터의 특징 벡터를 추출하고, 대형 언어 모델(LLM)을 통해 추가로 특징을 변환합니다.
	    ### 3.	SVM 분류기 학습: LLM에서 변환된 특징 벡터를 사용하여 SVM 분류기를 학습시킵니다.
        #features = self.encode_window(X, window = 8, batch_size = 2, window_batch_size = 2)
        sequence_features = self.encode_sequence(X, batch_size=2)
        score = numpy.mean(sklearn.model_selection.cross_val_score(
                    self.classifier, sequence_features, y=y, cv=5, n_jobs=5
                ))
        print("sequence encoder score:" , score)
        print("after trained sequence_features :", sequence_features[0])
        
        #print("feature's shape:", features.shape)
        print("feature's shape:", sequence_features.shape)
        
        # 지역별 인코딩된 시계열 데이터를 LLM에서 추가적으로 특징 변환(배치 처리)
        # features_from_llm_local = self.process_features_window(
        #     features, 
        #     batch_size=256, 
        # )
        #features_from_llm=self.llm(features)
        # 시점별 인코딩된 시계열 데이터를 LLM에서 추가적으로 특징 변환
        features_from_llm_sequence = self.process_features_window(
            sequence_features, 
            batch_size=256, 
        )
        
        print("after trained llm_features", features_from_llm_sequence[0])
        self.classifier = self.fit_classifier(features_from_llm_sequence, y)
        
        # features_from_llm과 y를 CPU로 이동하고 numpy 배열로 변환
        if isinstance(features_from_llm_sequence, torch.Tensor):
            features_from_llm_sequence = features_from_llm_sequence.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        
        # Cross validation score
        score = numpy.mean(sklearn.model_selection.cross_val_score(
            self.classifier, features_from_llm_sequence, y=y, cv=5, n_jobs=5
        ))
        print("features_from_llm_sequence score:" , score)
        
        # 시각화
        #self.plot_embeddings_2d(features, title="3D Features from encoder (local)", labels=y, method='tsne', save_path = '../visualization/encoder_local.png')
        #self.plot_embeddings_2d(features_from_llm_local, title="3D Features from LLM (local)", labels=y, method='tsne', save_path = '../visualization/llm_local.png')
        self.plot_embeddings_2d(sequence_features, title="3D Features from encoder (sequence)", labels=y, method='tsne', save_path = '../visualization/encoder_sequence.png')
        self.plot_embeddings_2d(features_from_llm_sequence, title="3D Features from LLM (sequence)", labels=y, method='tsne', save_path = '../visualization/llm_sequence.png')
        
        # Matching TS Embedding to Word(sequence)
        # matched_dict_local = self.semantic_matching(
        #     embeddings=features_from_llm_local,  # 또는 features
        #     X = X, # 원본 시계열 데이터
        #     labels=y,  # 클래스 레이블
        #     title="Semantic Mapping of Class Representatives(local)",
        #     save_path='../visualization_semantic/class_representatives(local).png'
        # )
        
        # Matching TS Embedding to Word(sequence)
        matched_dict_sequence = self.semantic_matching(
            embeddings=features_from_llm_sequence,  # 또는 features
            X = X, # 원본 시계열 데이터
            labels=y,  # 클래스 레이블
            title="Semantic Mapping of Class Representatives(sequence)",
            save_path='../visualization_semantic/class_representatives(sequence).png'
        )

        # # 4. 결과 확인
        # print("\nMatched words for each class(local):")
        # for word in matched_dict_local:
        #     print(f" {word}")
        
        print("\nMatched words for each class(sequence):")
        for word in matched_dict_sequence:
            print(f" {word}")

        return self

    # def encode(self, X, batch_size=2):
    #     # CausalCNN으로 특징 추출
    #     causal_features = super().encode(X, batch_size)  # 부모 클래스의 encode 호출
        
    #     # Data 객체 업데이트
    #     self.data.feature_df = causal_features
        
    #     # GPT2FeatureExtractor에 특징 전달
    #     llm_features = self.llm(torch.FloatTensor(causal_features))
        
    #     return llm_features.detach().cpu().numpy()
    
    def encode(self, X, batch_size=50):
        """
        Outputs the representations associated to the input by the encoder.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
            avoid out of memory errors when using CUDA. Ignored if the
            testing set contains time series of unequal lengths.
        """
        # Set device to GPU if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        # Convert input to PyTorch Dataset
        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )

        # Initialize output feature array
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))

        # Move encoder to the appropriate device
        self.encoder = self.encoder.to(device)
        self.encoder = self.encoder.eval()

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    # Move batch to device
                    batch = batch.to(device)

                    # Compute encoder output and move results back to CPU
                    features[
                        count * batch_size: (count + 1) * batch_size
                    ] = self.encoder(batch).cpu().numpy()
                    count += 1
            else:
                for batch in test_generator:
                    # Move batch to device
                    batch = batch.to(device)

                    # Handle variable-length sequences
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    features[count: count + 1] = self.encoder(
                        batch[:, :, :length]
                    ).cpu().numpy()
                    count += 1

        # Restore encoder to training mode
        self.encoder = self.encoder.train()

        return features

    def encode_window(self, X, window, batch_size=50, window_batch_size=10000):
        #### 하나의 시계열데이터에 대해 시계열 구간별로 여러 특징 벡터들을 뽑아냄.#####
        ##### 각 구간에 대해 개별적인 특징 벡터를 반환 ######
        """
        Outputs the representations associated to the input by the encoder,
        for each subseries of the input of the given size (sliding window
        representations).

        @param X Testing set.
        @param window Size of the sliding window.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA.
        @param window_batch_size Size of batches of windows to compute in a
               run of encode, to save RAM.
        """
        set_seed(self.seed)
        
        features = numpy.empty(( # (# of data_sample, feature dimension, 슬라이딩 윈도우로 생성된 시계열 구간 개수)
                numpy.shape(X)[0], self.out_channels,
                numpy.shape(X)[2] - window + 1
        ))
        masking = numpy.empty(( # 구간별 데이터 임시 저장 - (한 번에 처리할 구간 수, 시계열 채널 수, 윈도우 크기)
            min(window_batch_size, numpy.shape(X)[2] - window + 1),
            numpy.shape(X)[1], window
        ))
        for b in range(numpy.shape(X)[0]): # 샘플별 처리
            for i in range(math.ceil(   # 윈도우 가간을 배치로 처리
                (numpy.shape(X)[2] - window + 1) / window_batch_size)
            ):
                for j in range( # 구간별 슬라이딩 윈도우 처리
                    i * window_batch_size,
                    min(
                        (i + 1) * window_batch_size,
                        numpy.shape(X)[2] - window + 1
                    )
                ):
                    j0 = j - i * window_batch_size
                    masking[j0, :, :] = X[b, :, j: j + window]
                features[
                    b, :, i * window_batch_size: (i + 1) * window_batch_size
                ] = numpy.swapaxes(
                    self.encode(masking[:j0 + 1], batch_size=batch_size), 0, 1
                )
        return features
    
    # encode_window 메서드 다음에 추가
    def init_gpt2_embeddings(self):
        """GPT-2 모델과 토크나이저 초기화"""
        self.model_name = "gpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.gpt2_model = GPT2Model.from_pretrained(self.model_name)
        
        # pad_token 설정
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.cuda:
            self.gpt2_model = self.gpt2_model.cuda(self.gpu)

    def get_vocab_embeddings(self, batch_size=128):
        """배치 처리로 GPT-2 어휘 임베딩 생성"""
        vocab_size = self.tokenizer.vocab_size
        vocab_embeddings = []
        vocab_words = []

        for i in range(0, vocab_size, batch_size):
            tokens = list(range(i, min(i + batch_size, vocab_size)))
            words = [self.tokenizer.decode([token]) for token in tokens]
            vocab_words.extend(words)

            inputs = self.tokenizer(words, return_tensors="pt", padding=True, truncation=True)
            if self.cuda:
                inputs = {key: val.cuda(self.gpu) for key, val in inputs.items()}

            with torch.no_grad():
                outputs = self.gpt2_model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            vocab_embeddings.append(batch_embeddings)

        return vocab_words, numpy.vstack(vocab_embeddings)

    def save_embeddings(self, vocab_words, vocab_embeddings, file_prefix="gpt2_vocab"):
        """임베딩을 파일에 저장"""
        os.makedirs("embeddings", exist_ok=True)
        numpy.save(f"embeddings/{file_prefix}_words.npy", vocab_words)
        numpy.save(f"embeddings/{file_prefix}_embeddings.npy", vocab_embeddings)
        print(f"Embeddings saved to embeddings/{file_prefix}_words.npy and embeddings/{file_prefix}_embeddings.npy")

    def load_embeddings(self, file_prefix="gpt2_vocab"):
        """임베딩을 파일에서 불러오기"""
        vocab_words = numpy.load(f"embeddings/{file_prefix}_words.npy", allow_pickle=True)
        vocab_embeddings = numpy.load(f"embeddings/{file_prefix}_embeddings.npy")
        print(f"Embeddings loaded from embeddings/{file_prefix}_words.npy and embeddings/{file_prefix}_embeddings.npy")
        return vocab_words, vocab_embeddings

    def initialize_embeddings(self):
        """임베딩 초기화 또는 로드"""
        set_seed(self.seed)
        self.init_gpt2_embeddings()
        
        if not os.path.exists("embeddings/gpt2_vocab_embeddings.npy"):
            vocab_words, vocab_embeddings = self.get_vocab_embeddings()
            self.save_embeddings(vocab_words, vocab_embeddings)
        else:
            vocab_words, vocab_embeddings = self.load_embeddings()
        
        self.vocab_words = vocab_words
        self.vocab_embeddings = vocab_embeddings

    def predict(self, X, batch_size=50):
        """
        Outputs the class predictions for the given test data.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        features = self.encode(X, batch_size=batch_size)
        return self.classifier.predict(features)

    def score(self, X, y, batch_size=50):
        """
        Outputs accuracy of the SVM classifier on the given testing data.

        @param X Testing set.
        @param y Testing labels.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        features = self.encode(X, batch_size=batch_size)
        #features = self.encode_window(X, window = 8, batch_size = 2, window_batch_size = 2)
        print("feature's shape (method):", features.shape)
        # 인코딩된 시계열 데이터를 LLM에서 추가적으로 특징 변환(배치 처리)
        features_from_llm = self.process_features_window(
            features, 
            batch_size=256, 
        )
        return self.classifier.score(features_from_llm, y)

#####################################################
###### TimeSeriesEncoderLLM : 시계열 인코딩 후 LLM 활용, 더 복잡한 패턴 학습 및 언어 모델과의 통합, LLM으로 더 높은 수준의 표현 학습
###### CausalCNNEncoderLLM : Causal CNN을 사용한 시계열 인코딩, 시계열 데이터의 시간적 의존성과 인과관계 학습, 시간 순서를 기반으로 한 로컬 패턴 학습
######################################################

class CausalCNNEncoderLLM(TimeSeriesEncoderLLM):
    """
    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of randomly chosen intervals to select the
           final negative sample in the loss.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    @param batch_size Batch size used during the training of the encoder.
    @param nb_steps Number of optimization steps to perform for the training of
           the encoder.
    @param lr learning rate of the Adam optimizer used to train the encoder.
    @param penalty Penalty term for the SVM classifier. If None and if the
           number of samples is high enough, performs a hyperparameter search
           to find a suitable constant.
    @param early_stopping Enables, if not None, early stopping heuristic
           for the training of the representations, based on the final
           score. Representations are still learned unsupervisedly in this
           case. If the number of samples per class is no more than 10,
           disables this heuristic. If not None, accepts an integer
           representing the patience of the early stopping strategy.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of features in the final output.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param in_channels Number of input channels of the time series.
    @param cuda Transfers, if True, all computations to the GPU.
    @param gpu GPU index to use, if CUDA is enabled.
    """
    def __init__(self, compared_length=50, nb_random_samples=10,
                 negative_penalty=1, batch_size=1, nb_steps=2000, lr=0.001,
                 penalty=1, early_stopping=None, channels=10, depth=1,
                 reduced_size=10, out_channels=10, kernel_size=4,
                 in_channels=1, text_prototype = True, cuda=False, gpu=0,local_rank=-1):
        super(CausalCNNEncoderLLM, self).__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping,
            self.__create_encoder(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu,local_rank),
            self.__encoder_params(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size),
            in_channels, out_channels, text_prototype, cuda, gpu,local_rank   # text_prototpye 추가 
        )
        self.architecture = 'CausalCNN'
        self.channels = channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.kernel_size = kernel_size

    def __create_encoder(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu,local_rank):

        encoder = networks.causal_cnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        encoder.double()
        if cuda:
            if local_rank != -1:
                torch.cuda.set_device(local_rank)
                self.dist=torch.distributed.init_process_group(backend='nccl')
                device = torch.device('cuda', local_rank)
                encoder = encoder.cuda(device)
                encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[local_rank])
            else:
                encoder = encoder.cuda(gpu)

        return encoder

    def __encoder_params(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size):
        return {
            'in_channels': in_channels,
            'channels': channels,
            'depth': depth,
            'reduced_size': reduced_size,
            'out_channels': out_channels,
            'kernel_size': kernel_size
        }

    def encode_sequence(self, X, batch_size=50):
        """
        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
            avoid out of memory errors when using CUDA. Ignored if the
            testing set contains time series of unequal lengths.
        output : (n_samples, out_channels, sequence_length)
        """
        set_seed(self.seed)

        # GPU 사용 여부 명시적으로 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        # Convert dataset to PyTorch Dataset
        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test,
            batch_size=batch_size if not varying else 1,  # Handle unequal lengths
            shuffle=False,
            worker_init_fn=lambda worker_id: set_seed(self.seed + worker_id),
            generator=torch.Generator().manual_seed(self.seed)
        )

        # Initialize output features array
        length = numpy.shape(X)[2]
        features = numpy.full(
            (numpy.shape(X)[0], self.out_channels, length), numpy.nan
        )

        # Move encoder to device (GPU or CPU)
        self.encoder = self.encoder.to(device)
        self.encoder = self.encoder.eval()

        # Access submodules
        causal_cnn = self.encoder.network[0]
        linear = self.encoder.network[3]

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    # Move batch to device
                    batch = batch.to(device)
                    print(f"Batch device: {batch.device}")

                    # Apply causal CNN
                    output_causal_cnn = causal_cnn(batch)
                    after_pool = torch.empty(
                        output_causal_cnn.size(), dtype=torch.double, device=device
                    )
                    after_pool[:, :, 0] = output_causal_cnn[:, :, 0]

                    # Compute max pooling for each time step
                    for i in range(1, length):
                        after_pool[:, :, i] = torch.max(
                            torch.cat([
                                after_pool[:, :, i - 1: i],
                                output_causal_cnn[:, :, i: i+1]
                            ], dim=2),
                            dim=2
                        )[0]

                    # Compute linear transformation and store results
                    features[
                        count * batch_size: (count + 1) * batch_size, :, :
                    ] = torch.transpose(linear(
                        torch.transpose(after_pool, 1, 2)
                    ), 1, 2).cpu().numpy()  # Move to CPU and convert to numpy
                    count += 1
            else:
                for batch in test_generator:
                    batch = batch.to(device)
                    print(f"Batch device: {batch.device}")

                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    output_causal_cnn = causal_cnn(batch)
                    after_pool = torch.empty(
                        output_causal_cnn.size(), dtype=torch.double, device=device
                    )
                    after_pool[:, :, 0] = output_causal_cnn[:, :, 0]

                    for i in range(1, length):
                        after_pool[:, :, i] = torch.max(
                            torch.cat([
                                after_pool[:, :, i - 1: i],
                                output_causal_cnn[:, :, i: i+1]
                            ], dim=2),
                            dim=2
                        )[0]

                    features[
                        count: count + 1, :, :
                    ] = torch.transpose(linear(
                        torch.transpose(after_pool, 1, 2)
                    ), 1, 2).cpu().numpy()  # Move to CPU and convert to numpy
                    count += 1

        # Restore encoder to training mode
        self.encoder = self.encoder.train()
        return features

    ### 현재 모델의 모든 파라미터를 딕셔너리 형식으로 반환. encoder/classifier의 분류기 설정 값 모두 반환. 
    def get_params(self, deep=True):
        return {
            'compared_length': self.loss.compared_length,
            'nb_random_samples': self.loss.nb_random_samples,
            'negative_penalty': self.loss.negative_penalty,
            'batch_size': self.batch_size,
            'nb_steps': self.nb_steps,
            'lr': self.lr,
            'penalty': self.penalty,
            'early_stopping': self.early_stopping,
            'channels': self.channels,
            'depth': self.depth,
            'reduced_size': self.reduced_size,
            'kernel_size': self.kernel_size,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'cuda': self.cuda,
            'gpu': self.gpu,
            'local_rank':self.local_rank
        }

    def set_params(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu,local_rank):
        self.__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels,cuda, gpu,local_rank
        )
        return self
