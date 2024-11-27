from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model
from einops import rearrange
from networks.embedding import DataEmbedding
# from pip import get_peft_config, get_peft_model, LoraConfig, TaskType


class GPT2FeatureExtractor(nn.Module):

    def __init__(self, config, data):
        super(GPT2FeatureExtractor, self).__init__()
        self.pred_len = 0
        self.seq_len = config.get('seq_len', data.max_seq_len) #data.max_seq_len
        print("seq_len: ", self.seq_len)   #128
        #self.max_len = self.seq_len #data.max_seq_len
        self.patch_size = config['patch_size']   #16
        self.stride = config['stride']   #8
        self.gpt_layers = 6
        self.feat_dim = config.get('feature_dim', 768) #data.feature_df.shape[1]
        # 클래스 수를 config에서 가져오도록 수정
        #self.num_classes = config.get('num_classes', 2)  # 기본값 2로 설정
        self.num_classes = 7 #len(data.class_names)
        self.d_model = config['d_model']
        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1
        
        self.required_padding = (self.stride - (self.seq_len - self.patch_size) % self.stride) % self.stride
        print(f"Calculated padding size: {self.required_padding}")
        self.patch_num += 1 if self.required_padding > 0 else 0  # 추가 패치가 필요한 경우 패치 수 증가

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.required_padding))

        #self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        #self.patch_num += 1
        self.enc_embedding = DataEmbedding(self.feat_dim * self.patch_size, config['d_model'], config['dropout'])

        # GPT-2 모델을 Hugging Face Hub에서 다운로드하여 사용
        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        
        # GPT-2 레이어 제한 (처음 6개의 레이어만 사용)
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]

        # 파라미터 고정 (LayerNorm 및 positional embeddings을 제외한 나머지는 고정)
        for name, param in self.gpt2.named_parameters():
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gpt2.to(self.device)

        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.ln_proj = nn.LayerNorm(config['d_model'] * self.patch_num).to(self.device)
        print(f"d_model: {config['d_model']}, patch_num: {self.patch_num}, num_classes: {self.num_classes}")
        self.out_layer = nn.Linear(config['d_model'] * self.patch_num, self.num_classes).to(self.device)

        # 모델 전체를 동일한 장치로 이동
        self.to(self.device)

    def forward(self, x_enc):
        # Ensure x_enc is a tensor
        if isinstance(x_enc, np.ndarray):
            x_enc = torch.tensor(x_enc)
        
        # Ensure x_enc is on the same device as the model
        x_enc = x_enc.to(self.device)
        
        B, L, M = x_enc.shape # [batch_size, sequence_length, model_dimension]
    
        # 특징 차원으로 feat_dim 업데이트
        self.feat_dim = M
        self.seq_len = L
        
        # 패딩이 필요한 경우를 고려한 패치 수 계산
        if (self.seq_len - self.patch_size) % self.stride != 0:
            # 마지막 패치를 위한 패딩이 필요한 경우
            self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1
        else:
            # 패딩이 필요없는 경우
            self.patch_num = (self.seq_len - self.patch_size) // self.stride
            
        self.patch_num += 1
        
        # 레이어 재초기화
        self.ln_proj = nn.LayerNorm(self.d_model * self.patch_num).to(self.device)
        print(f"d_model: {self.d_model}, patch_num: {self.patch_num}, num_classes: {self.num_classes}")
        self.out_layer = nn.Linear(self.d_model * self.patch_num, self.num_classes).to(self.device)    ##llm_wrappers의 fit 메서드에서 자동 계산해서 class 개수 업데이트 
        
        # DataEmbedding 재초기화 (필요한 경우)
        if not hasattr(self, 'enc_embedding') or self.enc_embedding.value_embedding.tokenConv.in_channels != L * self.patch_size:
            self.enc_embedding = DataEmbedding(
                c_in=M * self.patch_size,
                d_model=self.d_model,
                dropout=self.dropout.p
            ).to(x_enc.device)

        # Ensure input_x is a tensor for padding operations
        input_x = rearrange(x_enc, 'b l m -> b m l').to(self.device)  # [256, 1024, 96]
        print(f"Before padding: {input_x.shape}")  # [batch_size, n_channels, sequence_length]
        input_x = self.padding_patch_layer(input_x)   # [256, 1024, 104]
        print(f"After padding: {input_x.shape}")  # [batch_size, n_channels, sequence_length + padding]
        # 패치 생성
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # [256, 1024, 13, 8]
        print(f"After unfolding: {input_x.shape}")
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')  # shape: [256, 13, 8192]    # 8192 = 1024(features) * 8(patch_size)
        print(f"After rearranging for embedding: {input_x.shape}")

        # GPT-2에 입력하기 전에 임베딩
        outputs = self.enc_embedding(input_x.to(self.device), None)  # [256, 13, 768]
        print(f"Embedding output shape: {outputs.shape}")

        # GPT-2 모델을 통해 추론
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state
        print("outputs'shape: ", outputs.shape)  # [256, 11, 768]
        print("patch_num : ", self.patch_num)

        # 마지막 레이어 처리
        outputs = self.act(outputs).reshape(B, -1)
        print(f"After activation and reshaping: {outputs.shape}")    #[256, 8448]

        outputs = self.ln_proj(outputs)  # self.device로 이동된 LayerNorm
        #outputs = self.out_layer(outputs)  # self.device로 이동된 Linear 레이어

        return outputs