import os
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./configs/LLM1B.py")
args = parser.parse_args()

def prepare_torchrun_dist():
    GPU_PER_NODE = 8
    NNODES = 1
    NODE_RANK = 0
    MASTER_ADDR = '127.0.0.1'
    MASTER_PORT = '1234'

    DIST_ARGS = f'--nproc_per_node {GPU_PER_NODE} --nnodes {NNODES} --node_rank {NODE_RANK} --master_addr {MASTER_ADDR} --master_port {MASTER_PORT} --rdzv_conf timeout=5400'

    return DIST_ARGS

dist_args = prepare_torchrun_dist()

command = (f'torchrun {dist_args} '
           f'train.py '
           f'--config {args.config} '
           f'--launcher "torch" '
           )
os.system(f'{command}')