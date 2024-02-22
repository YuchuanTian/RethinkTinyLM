import os
import json
from transformers import LlamaTokenizer, AutoTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import argparse
from tqdm import tqdm

def is_special_token(token):
    return ((token.startswith('<') and token.endswith('>') and len(token) > 2) or
            (token.startswith('[') and token.endswith(']') and len(token) > 2))

parser = argparse.ArgumentParser()
parser.add_argument('--origin_tokenizer_dir', type=str, required=True)
parser.add_argument('--vocab_num', type=int, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--token_freq_file', type=str, required=True)
args = parser.parse_args()

# origin_tokenizer
origin_tokenizer = LlamaTokenizer.from_pretrained(args.origin_tokenizer_dir, trust_remote_code=True)
llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(origin_tokenizer.sp_model.serialized_model_proto())

vocabs = {v:k for k,v in origin_tokenizer.get_vocab().items()}

with open(args.token_freq_file, "r") as f:
    d = json.load(f)

for k,v in vocabs.items():
    if str(k) not in d:
        d[str(k)] = 0

sorted_d = sorted(d.items(), key=lambda x: x[1], reverse=True)

remained = set() # remain vocab
# keep the special tokens
for k,v in sorted_d:
    if is_special_token(vocabs[int(k)]):
        remained.add(vocabs[int(k)])

# keep the top frequency tokens
for k,v in tqdm(sorted_d):
    if vocabs[int(k)] not in remained:
        remained.add(vocabs[int(k)])
    if len(remained) >= args.vocab_num:
        break

cnt = 0
removed_pieces = []
for piece in llama_spm.pieces:
    if piece.piece in remained:
        cnt += 1
    else:
        removed_pieces.append(piece)
assert cnt == args.vocab_num

# remove tokens with low frequency
for piece in tqdm(removed_pieces):
    llama_spm.pieces.remove(piece)

print(f'Final length of vocab: {len(llama_spm.pieces)}')

## Save
if not os.path.exists(args.output):
    os.makedirs(args.output, exist_ok=True)
model_file = os.path.join(args.output, 'tokenizer.model')
with open(model_file, 'wb') as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=model_file)
tokenizer.save_pretrained(args.output)
