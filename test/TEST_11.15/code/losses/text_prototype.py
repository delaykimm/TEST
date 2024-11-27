import os
import random
import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import GPT2Model, GPT2Tokenizer
import argparse

def select_representative(embedding_matrix, k):
    m, n = embedding_matrix.shape
    result = np.empty(m, dtype=int)
    cores = embedding_matrix[np.random.choice(np.arange(m), k, replace=False)]
    while True:
        d = np.square(np.repeat(embedding_matrix, k, axis=0).reshape(m, k, n) - cores)
        distance = np.sqrt(np.sum(d, axis=2))
        index_min = np.argmin(distance, axis=1)
        if (index_min == result).all():
            return cores

        print('[{}/{}]'.format(sum(index_min == result),m))
        result[:] = index_min
        for i in range(k):
            items = embedding_matrix[result == i]
            cores[i] = np.mean(items, axis=0)
    return cores

def select_prototype(model_name='gpt2', prototype_dir=None, provide=False, number_of_prototype=10):
    # 시드값 고정
    seed = 42
    random.seed(seed)  # Python random 시드 고정
    torch.manual_seed(seed)  # PyTorch 시드 고정 (CPU)
    torch.cuda.manual_seed(seed)  # PyTorch 시드 고정 (GPU 사용 시)

    # GPT-2 모델과 토크나이저 불러오기
    print(f'Loading GPT-2 model: {model_name}')
    gpt2_model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # GPT-2 임베딩 매트릭스 추출 (word embeddings: 'wte.weight')
    embedding_matrix = gpt2_model.get_input_embeddings().weight  # GPT-2에서 임베딩 매트릭스를 추출

    if isinstance(provide, list):
        print('----Select provided text prototypes----')
        # 제공된 텍스트를 토큰화하고 임베딩을 추출
        tokens = tokenizer.tokenize(''.join(provide))
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        embeddings = embedding_matrix[token_ids]
        if prototype_dir:
            torch.save(embeddings, os.path.join(prototype_dir, 'text_prototype_provided.pt'))

    elif provide == 'random':
        print(f'----Select {number_of_prototype} random text prototypes----')
        # 무작위로 토큰을 선택하고 해당하는 임베딩을 추출
        token_ids = [random.randint(0, len(embedding_matrix) - 1) for i in range(number_of_prototype)]
        embeddings = embedding_matrix[token_ids].clone()
        if prototype_dir:
            torch.save(embeddings, os.path.join(prototype_dir, 'text_prototype_random.pt'))

    else:
        print(f'----Select {number_of_prototype} representative text prototypes----')
        # 임베딩을 numpy 배열로 변환한 후 대표 임베딩 선택 (여기서는 select_representative 함수가 필요함)
        embedding_matrix = embedding_matrix.detach().cpu().numpy()
        embeddings = select_representative(embedding_matrix, number_of_prototype)
        embeddings = torch.from_numpy(embeddings)
        if prototype_dir:
            torch.save(embeddings, os.path.join(prototype_dir, 'text_prototype_representative.pt'))

    return embeddings, embeddings.size()

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Prototype selection from LLM downloaded from huggingface'
    )
    parser.add_argument('--llm_model_dir', type=str, metavar='PATH', required=False,
                        help='path where the LLM is located')
    parser.add_argument('--prototype_dir', type=str, metavar='PATH', required=True,
                        help='path where the prototype file to save')
    parser.add_argument('--provide', type=str, default=False,
                        help='Provide or select the prototypes. (random, False, True)')
    parser.add_argument('--number_of_prototype', type=int, default=10, metavar='Number',
                        help='Number of prototype to select')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    #embeddings = select_prototype(model_dir="../../gpt2", prototype_dir='/gpt2_protptype',provide='random', number_of_prototype=10)
    #embeddings = select_prototype(model_dir="../../gpt2", prototype_dir='/gpt2_protptype',provide=['value','shape','frequency'])
    # embeddings = select_prototype(model_dir="../../gpt2", prototype_dir='/gpt2_protptype',provide=False,number_of_prototype=10)
    #embeddings,size=select_prototype(prototype_dir=args.prototype_dir,provide=args.provide,number_of_prototype=args.number_of_prototype)
    print(size)

