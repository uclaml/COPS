import os
import json
import argparse
import numpy as np
import math
from models import online_embed, build_server
import faiss
import logging
import random

import lats
from lats import dfs_search,quote_env
from webshop import WebShopTask


def random_selection(lst, n=5):
    if len(lst) <= n:
        return lst
    else:
        return random.sample(lst, n)


def run(args):
    task = WebShopTask()
    print(task)
    logs, cnt_avg, cnt_any = [], 0, 0

    # create log directories if they don't exist
    config_path = os.path.join(args.run_name, "config.json")

    with open(config_path, "w", encoding="utf-8") as wf:
        info_dict = vars(args)
        info_dict["is_running"] = True
        json.dump(info_dict, wf, indent=4)

    build_server(config_path=config_path)

    os.makedirs(os.path.dirname(args.log_dir), exist_ok=True)

    logging.basicConfig(
        filename=args.log_dir,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",
    )

    wins = {}
    lose = {}
    trajectories = []
    embedding_array = np.zeros((0, 3584))
    tongji=[]
    avg_tongji=[]
    cnt=0
    scores=[]
    for i in range(50):
        scores.append(0.0)
    for trial in range(10):
        with open("debug_log.log", "a", encoding="utf-8") as f:
            f.write("Trial")
            f.write(str(trial)+"\n")
        count = 0
        task_accs = []
        info = []
        emb_db = faiss.IndexFlatL2(3584)
        emb_db.add(embedding_array.astype("float32"))
        avg=0.0
        for i in range(args.task_start_index, args.task_end_index):
            with open("debug_log1.log", "a", encoding="utf-8") as f:
                f.write("------------new task---------------\n")
                f.write(str(i)+"\n")
            # solve
            if i in wins:
                continue
            prev = None
            knnret = []
            if trajectories and args.cot_size>0:
                if i in lose:
                    prev = lose[i]
                else:
                    prev=quote_env(f'fixed_{i}')
                if args.cot_method=="knn":
                    fail_vec = online_embed(str(prev))
                    dist=[]
                    for index, row in np.ndenumerate(embedding_array):
                        if index[1] == 0:
                            realN = (1.0 / np.linalg.norm(np.array(fail_vec) - embedding_array[index[0]]))
                            dist.append((realN,index[0]))
                    dist.sort(key=lambda x: x[0], reverse=True)
                    sz_now=min(len(trajectories),args.cot_size)
                    dist=dist[:sz_now]
                    dist=[(math.exp(5.0*x),y) for (x,y) in dist]
                    original_sum=sum([x for (x,_) in dist])
                    dist=[(x *float(sz_now) / original_sum,y) for (x,y) in dist]
                    tot=0
                    cntD=[]
                    realD=[]
                    for (x,y) in dist:
                        cntD.append(math.floor(x))
                        realD.append(x)
                        tot+=math.floor(x)
                    while tot<sz_now:
                        maxx=-0.1
                        maxi=-1
                        for ig in range(len(cntD)):
                            if realD[ig]-float(cntD[ig])>maxx:
                                maxx=realD[ig]-float(cntD[ig])
                                maxi=ig
                        cntD[maxi]+=1
                        tot+=1
                    weights=[]
                    for (x,y) in dist:
                        weights.append(x)
                    normalized_weights = [w / sum(weights) for w in weights]
                    for ig in range(sz_now):
                        s_ind = np.random.choice(len(weights), p=normalized_weights)
                        (_,y)=dist[s_ind]
                        knnret.append(trajectories[y])
                else:
                    knnret=random.sample(trajectories, min(args.cot_size,len(trajectories)))
            state, value, all_nodes, reward, em, failt, succt = dfs_search(
                args, task, f'fixed_{i}', args.iteration, knnret
            )
            if failt:
                with open("debug_log.log", "a", encoding="utf-8") as f:
                    f.write("FAIL\n")
                print("Fail")
                print(i)
                lose[i] = failt[0]
            if succt:
                cnt=cnt+1
                with open("debug_log.log", "a", encoding="utf-8") as f:
                    f.write("SUCCESS\n")
                print("Success")
                print(i)
                wins[i] = 1
                vec = online_embed(str(succt[0]))
                trajectories.append(succt[0])
                embedding_array = np.vstack((embedding_array, np.array(vec)))
            scores[i]=max(scores[i],reward)
            avg+=reward
            # log main metric
            if em is None:
                em = 0
            task_accs.append(em)
            cnt_avg = sum(task_accs) / len(task_accs)
            print(i, "len(task_accs)", len(task_accs), "cnt_avg", cnt_avg, "\n")
        tongji.append(cnt)
        avg=sum(scores)/50.0
        with open("debug_log.log", "a", encoding="utf-8") as f:
            f.write("average: ")
            f.write(str(avg)+"\n")
        avg_tongji.append(avg)
    print("prompt:")
    print(lats.prompt_tokens)
    print("completion:")
    print(lats.completion_tokens)
    print(tongji)
    print(avg_tongji)
    n = args.task_end_index - args.task_start_index


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model_size", type=str, default="8")
    args.add_argument("--temperature", type=float, default=1.0)
    args.add_argument("--task_start_index", type=int, default=900)
    args.add_argument("--task_end_index", type=int, default=1000)
    args.add_argument("--prompt_sample", type=str, choices=["standard", "cot"])
    args.add_argument("--n_generate_sample", type=int, default=1)
    args.add_argument("--n_evaluate_sample", type=int, default=1)
    args.add_argument("--iteration", type=int, default=50)
    args.add_argument("--algorithm", type=str, choices=["lats", "rap", "tot"])
    args.add_argument("--cot_method", type=str, choices=["knn", "random", "None"])
    args.add_argument("--run_name", type=str)
    args.add_argument("--log_file_path", type=str)
    args.add_argument("--log_dir", type=str)
    args.add_argument("--cot_size", type=int, default=0)
    args = args.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    run(args)
