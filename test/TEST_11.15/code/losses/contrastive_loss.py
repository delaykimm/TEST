import torch
import numpy
import random


class ContrastiveLoss(torch.nn.modules.loss._Loss):
    """
    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of negative samples per batch example.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    @param text_prototype If True, include text-prototype-aligned contrast loss.
    """
    def __init__(self, compared_length, nb_random_samples, negative_penalty, text_prototype = False, use_hard_negative=False, seed=42
                 ):
        super(ContrastiveLoss, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty
        self.text_prototype = text_prototype
        self.use_hard_negative = use_hard_negative
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_random_seed(seed)
        self._check_device()
        print("text_prototype :", self.text_prototype)
        print("use_hard_negative :", self.use_hard_negative) 
        print("seed :", self.seed)
        
    def _set_random_seed(self, seed):
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def _check_device(self):
        """
        Check if GPU is available and print device information.
        """
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
        else:
            print("Using CPU for computations.")

    def forward(self, batch, encoder, train, save_memory=False, text_prototype_embeddings = None):
        # Move all tensors to the same device
        batch = batch.to(self.device)
        train = train.to(self.device)
        encoder = encoder.to(self.device)
        if text_prototype_embeddings is not None:
            text_prototype_embeddings = text_prototype_embeddings.to(self.device)
        
        print(text_prototype_embeddings.shape)
        batch_size = batch.size(0)
        train_size = train.size(0)
        length = min(self.compared_length, train.size(2))

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        
        # Negative sampling with hard negative mining (if enabled)
        if self.use_hard_negative:
            with torch.no_grad():
                all_train_representations = encoder(train)
                similarities = torch.matmul(
                    encoder(batch).detach(), all_train_representations.T)
                top_k_indices = torch.topk(similarities, self.nb_random_samples, dim=1)[1]
                samples = top_k_indices.cpu().numpy()
        else:
            samples = torch.tensor(
                numpy.random.choice(train_size, size=(self.nb_random_samples, batch_size)),
                dtype = torch.long,
                device = self.device 
            )
        
        samples = numpy.random.choice(
            train_size, size=(self.nb_random_samples, batch_size)
        )
        samples = torch.LongTensor(samples)

        # Choice of length of positive and negative samples
        length_pos_neg = numpy.random.randint(low=1, high=length + 1)

        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = numpy.random.randint(
            low=length_pos_neg, high=length + 1
        )  # Length of anchors
        beginning_batches = numpy.random.randint(
            low=0, high=length - random_length + 1, size=batch_size
        )  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        beginning_samples_pos = numpy.random.randint(
            low=0, high=random_length - length_pos_neg + 1, size=batch_size
        )  # Start of positive samples in the anchors
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + length_pos_neg

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = numpy.random.randint(
            low=0, high=length - length_pos_neg + 1,
            size=(self.nb_random_samples, batch_size)
        )
        
        # Anchors representations
        representation = encoder(torch.cat(
            [batch[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length
            ] for j in range(batch_size)]
        ))

        # Positive samples representations 
        positive_representation = encoder(torch.cat(
            [batch[
                j: j + 1, :, end_positive[j] - length_pos_neg: end_positive[j]
            ] for j in range(batch_size)]
        ))

        size_representation = representation.size(1)
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations
        
        # 기본 contrastive loss 계산
        contrastive_loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                representation.view(batch_size, 1, size_representation),
                positive_representation.view(batch_size, size_representation, 1)
        )))
        
        # Detach contrastive_loss to avoid graph retention
        contrastive_loss_value = contrastive_loss.item()
        
        # 메모리 관리 : positive sample 관련 메모리 해제
        if save_memory:
            loss.backward()
            loss = 0
            del positive_representation
            torch.cuda.empty_cache()
        
        # Text prototype이 있는 경우 text-prototype-aligned contrast 추가
        if self.text_prototype and text_prototype_embeddings is not None:
            print("Training with text_prototype-aligned contrast")
            text_prototype_loss = 0
            num_prototypes = text_prototype_embeddings.size(0) # P개의 프로토타입 (text_prototype_embeddings: [P, size_representation])
            
            # Step 1: Text Alignment: Calculate similarity between TS embeddings and all text prototypes
            # Representation: [batch_size, size_representation]
            # Text prototypes: [P, size_representation]
            # Similarity: [batch_size, P]
            representation = representation.float()
            similarities = torch.matmul(representation, text_prototype_embeddings.T)  # Batch-wise similarity
            
            for i in range(num_prototypes): # Loop over each text prototype
                print("prototype : ", i)
                # # 1. Text Alignment 항
                # text_alignment = torch.bmm(
                #     representation.view(batch_size, 1, size_representation),
                #     tpi.view(1, size_representation, 1)
                # )
                # Current prototype vector
                tpi = text_prototype_embeddings[i]  # [size_representation]
                
                # Alignment term: Similarity between TS embedding and prototype
                text_alignment = similarities[:, i].view(batch_size, 1)  # Extract i-th prototype similarity for all batches
                
                # 2. Feature Contrast 항 
                e_tp = representation * tpi
                e_tp_pos = positive_representation * tpi if not save_memory else None
                
                if not save_memory:
                    e_tp_pos = e_tp_pos.float()
                    feature_pos = torch.bmm(
                        e_tp.view(batch_size, 1, size_representation),
                        e_tp_pos.view(batch_size, size_representation, 1)
                    )
                    feature_contrast_loss = -torch.mean(
                        torch.nn.functional.logsigmoid(feature_pos)
                    )
                #print(f"e_tp dtype: {e_tp.dtype}")
                #print(f"e_tp_pos dtype: {e_tp_pos.dtype}")
                
                # Negative samples 처리
                feature_neg_loss = 0
                multiplicative_ratio = self.negative_penalty / self.nb_random_samples
                
                for j in range(self.nb_random_samples): # Loop over negative samples
                    # Negative representation sampling
                    negative_representation = encoder(torch.cat(
                        [train[samples[j, k]: samples[j, k] + 1][
                            :, :,
                            beginning_samples_neg[j, k]:
                            beginning_samples_neg[j, k] + length_pos_neg
                        ] for k in range(batch_size)]
                    ))
                    
                    e_tp_neg = negative_representation * tpi
                    #print(f"e_tp dtype: {e_tp.dtype}")
                    #print(f"e_tp_neg dtype: {e_tp_neg.dtype}")
                    
                    e_tp_neg = e_tp_neg.float()
                    feature_neg = torch.bmm(
                        e_tp.view(batch_size, 1, size_representation),
                        e_tp_neg.view(batch_size, size_representation, 1)
                    )
                    feature_neg_loss += -torch.mean(
                        torch.nn.functional.logsigmoid(-feature_neg)
                    )
                    
                    # 메모리 관리: negative sample 관련 메모리 해제
                    if save_memory and j != self.nb_random_samples - 1:
                        feature_neg_loss.backward()
                        feature_neg_loss = 0
                        del negative_representation, e_tp_neg, feature_neg
                        torch.cuda.empty_cache()
                
                # Combine positive and negative losses
                feature_contrast_loss = (feature_contrast_loss if not save_memory else 0) + \
                    multiplicative_ratio * feature_neg_loss
                
                # Prototype loss for current prototype
                prototype_loss = -(torch.mean(text_alignment) - feature_contrast_loss)
                text_prototype_loss += prototype_loss.item()  # Detach from graph
            
            # Step 3: Normalize by the number of prototypes
            text_prototype_loss = text_prototype_loss / num_prototypes
            
            # Combine total loss
            total_loss = contrastive_loss + text_prototype_loss
            print("total_loss", total_loss)
            return total_loss
        
        # if self.text_prototype==False:
        #     loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
        #         representation.view(batch_size, 1, size_representation),
        #         positive_representation.view(batch_size, size_representation, 1)
        #     )))
        # else:
        #     save_memory=False
        #     loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
        #         representation.view(batch_size, 1, size_representation),
        #         positive_representation.view(batch_size, size_representation, 1)
        #     ))+torch.nn.functional.logsigmoid(torch.bmm(
        #         representation.view(batch_size, 1, size_representation),
        #         self.text_prototype.view(batch_size, size_representation, 1)))
        #                        )

        # 기존 Negative samples에 대한 contrastive loss 계산
        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            negative_representation = encoder(
                torch.cat([train[samples[i, j]: samples[i, j] + 1][
                    :, :,
                    beginning_samples_neg[i, j]:
                    beginning_samples_neg[i, j] + length_pos_neg
                ] for j in range(batch_size)])
            )
            contrastive_loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    negative_representation.view(
                        batch_size, size_representation, 1
                    )
                ))
            )
            
            # 메모리 관리 : negative sample 관련 메모리 해제
            if save_memory and i != self.nb_random_samples - 1:
                contrastive_loss.backward(retain_graph=True)
                contrastive_loss = 0
                del negative_representation
                torch.cuda.empty_cache()

        return contrastive_loss


class ContrastiveLossVaryingLength(torch.nn.modules.loss._Loss):
    """
    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of negative samples per batch example.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    """
    def __init__(self, compared_length, nb_random_samples, negative_penalty,text_prototype=False):
        super(ContrastiveLossVaryingLength, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty
        self.text_prototype = text_prototype

    def forward(self, batch, encoder, train, save_memory=False, text_prototype_embeddings = None):
        batch_size = batch.size(0)
        train_size = train.size(0)
        max_length = train.size(2)

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        samples = numpy.random.choice(
            train_size, size=(self.nb_random_samples, batch_size)
        )
        samples = torch.LongTensor(samples)

        # Computation of the lengths of the relevant time series
        with torch.no_grad():
            lengths_batch = max_length - torch.sum(
                torch.isnan(batch[:, 0]), 1
            ).data.cpu().numpy()
            lengths_samples = numpy.empty(
                (self.nb_random_samples, batch_size), dtype=int
            )
            for i in range(self.nb_random_samples):
                lengths_samples[i] = max_length - torch.sum(
                    torch.isnan(train[samples[i], 0]), 1
                ).data.cpu().numpy()

        # Choice of lengths of positive and negative samples
        lengths_pos = numpy.empty(batch_size, dtype=int)
        lengths_neg = numpy.empty(
            (self.nb_random_samples, batch_size), dtype=int
        )
        for j in range(batch_size):
            lengths_pos[j] = numpy.random.randint(
                1, high=min(self.compared_length, lengths_batch[j]) + 1
            )
            for i in range(self.nb_random_samples):
                lengths_neg[i, j] = numpy.random.randint(
                    1,
                    high=min(self.compared_length, lengths_samples[i, j]) + 1
                )

        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = numpy.array([numpy.random.randint(
            lengths_pos[j],
            high=min(self.compared_length, lengths_batch[j]) + 1
        ) for j in range(batch_size)])  # Length of anchors
        beginning_batches = numpy.array([numpy.random.randint(
            0, high=lengths_batch[j] - random_length[j] + 1
        ) for j in range(batch_size)])  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        # Start of positive samples in the anchors
        beginning_samples_pos = numpy.array([numpy.random.randint(
            0, high=random_length[j] - lengths_pos[j] + 1
        ) for j in range(batch_size)])
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + lengths_pos

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = numpy.array([[numpy.random.randint(
            0, high=lengths_samples[i, j] - lengths_neg[i, j] + 1
        ) for j in range(batch_size)] for i in range(self.nb_random_samples)])

        representation = torch.cat([encoder(
            batch[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length[j]
            ]
        ) for j in range(batch_size)])  # Anchors representations

        positive_representation = torch.cat([encoder(
            batch[
                j: j + 1, :,
                end_positive[j] - lengths_pos[j]: end_positive[j]
            ]
        ) for j in range(batch_size)])  # Positive samples representations

        size_representation = representation.size(1)
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations
        loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            representation.view(batch_size, 1, size_representation),
            positive_representation.view(batch_size, size_representation, 1)
        )))

        # If required, backward through the first computed term of the loss and
        # free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del positive_representation
            torch.cuda.empty_cache()

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            negative_representation = torch.cat([encoder(
                train[samples[i, j]: samples[i, j] + 1][
                    :, :,
                    beginning_samples_neg[i, j]:
                    beginning_samples_neg[i, j] + lengths_neg[i, j]
                ]
            ) for j in range(batch_size)])
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    negative_representation.view(
                        batch_size, size_representation, 1
                    )
                ))
            )
            # If required, backward through the first computed term of the loss
            # and free from the graph everything related to the negative sample
            # Leaves the last backward pass to the training procedure
            if save_memory and i != self.nb_random_samples - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del negative_representation
                torch.cuda.empty_cache()

        return loss

