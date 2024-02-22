import multiprocessing, os, json
import tqdm, argparse

def worker(ori_file, save_path):
    res = {"ori_file": ori_file}
    with open(ori_file, "r") as f:
        line = f.readline()
        while line:
            try:
                data = json.loads(line)
                for token in data["tokens"]:
                    if token in res:
                        res[token] += 1
                    else:
                        res[token] = 1
            except:
                pass    
            line = f.readline()
    with open(save_path, "w") as f:
        f.write(json.dumps(res))
    return None

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, type=str)
    parser.add_argument('--dst', required=True, type=str)
    parser.add_argument('--process_num', default=128, type=int)
    args = parser.parse_args()
    print(args)

    datasets = []
    for root, dirs, files in os.walk(args.src, followlinks=True):
        dirs.sort()
        for fn in sorted(files):
            if fn.endswith(".bin"):
                fp = os.path.join(root, fn)
                datasets.append(fp)
    with multiprocessing.Pool(processes=args.process_num) as pool:
        results = []
        for i,ori_file in enumerate(datasets):
            results.append(pool.apply_async(worker, (ori_file, os.path.join(args.dst, "{}.json".format(ori_file.replace('/', '__'))))))
        pool.close()
        for result in tqdm.tqdm(results):
            result.get()
        pool.join()

