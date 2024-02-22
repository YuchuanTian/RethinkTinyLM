import os, json
import tqdm, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src', required=True, type=str)
parser.add_argument('--dst', required=True, type=str)
args = parser.parse_args()
print(args)

src = args.src
res = {}
for file in tqdm.tqdm(os.listdir(src)):
    with open(os.path.join(src, file)) as f:
        t = json.load(f)
    for k, v in t.items():
        if k == 'ori_file':
            continue
        if k in res:
            res[k] += v
        else:
            res[k] = v
save_path = args.dst
with open(save_path, "w") as f:
    f.write(json.dumps(res))


