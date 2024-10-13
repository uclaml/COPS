"""Adapted from https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb"""

import os

os.environ["PYTHONUTF8"] = "1"
import sys
import json
import yaml
import importlib
import alfworld
import re
import numpy as np
import faiss
import random
import math
import alfworld.agents.environment
from utils import EnvironmentHistory, ModelServer
from IPython import embed
from typing import List, Dict, Any, Tuple


EMBEDDING_DIM = {"7": 3584, "2": 1536}

FOLDER = "./prompts"
PROMPT_FILE = "alfworld_3prompts.json"

with open(os.path.join(FOLDER, PROMPT_FILE), "r", encoding="utf-8") as f:
    d = json.load(f)


PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


def process_ob(ob):
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2 :]
    return ob


def alfworld_run(
    model_server: ModelServer,
    plan_model_size: str,
    env,
    base_prompt,
    memory: List[str],
    to_print=True,
    ob="",
    mem_size: int = 0,
    mem_selection_method: str = "fifo",
    config_path: str = None,
) -> Tuple[EnvironmentHistory, bool]:

    assert mem_selection_method in [
        "fifo",
        "fix",
        "random",
    ], "Invalid memory selection method"

    if mem_size == 0:
        env_history = EnvironmentHistory(base_prompt, ob, [], [])
    elif mem_size != -1 and len(memory) >= mem_size:
        if mem_selection_method == "fifo":
            env_history = EnvironmentHistory(base_prompt, ob, memory[-mem_size:], [])
        elif mem_selection_method == "fix":
            env_history = EnvironmentHistory(base_prompt, ob, memory[:mem_size], [])
        elif mem_selection_method == "random":
            env_history = EnvironmentHistory(
                base_prompt, ob, random.sample(memory, mem_size), []
            )
    else:
        env_history = EnvironmentHistory(base_prompt, ob, memory, [])

    env_history.reset()
    if to_print:
        print(ob)
        sys.stdout.flush()
    cur_step = 0
    preva = ""
    message = [{"role": "user", "content": str(env_history)}]
    while cur_step < 49:
        tmpr = 0.0
        action = preva
        while action == preva:
            print(f"cur_step: {cur_step}")
            print(f"temperature: {tmpr}")
            completion = model_server.get_completion_or_embedding(
                plan_model_size,
                message=message,
                temperature=tmpr,
            )
            lines = completion.split("\n")
            action = ""
            for line in lines:
                stripped_line = line.strip()
                if stripped_line != "":
                    action = stripped_line
                    break

            if ">" in action:
                action = action.replace(">", "").strip()
            action_words = action.split(" ")
            if "put" in action_words:
                for i in range(len(action_words)):
                    if (
                        action_words[i].strip().lower() == "in"
                        or action_words[i].strip().lower() == "on"
                    ):
                        action_words[i] = "in/on"
                        action = " ".join(action_words)
            tmpr += 0.1
        env_history.add("action", action)
        message.append({"role": "assistant", "content": action})
        preva = action
        observation, _, done, info = env.step([action])
        observation, _, done = process_ob(observation[0]), info["won"][0], done[0]
        if action.startswith("think:"):
            observation = "OK."
        env_history.add("observation", observation)
        message.append({"role": "user", "content": observation})
        if to_print:
            print(f"> {action}\n{observation}")
            sys.stdout.flush()
        if done:
            return env_history, True
        elif env_history.check_is_exhausted():
            return env_history, False
        cur_step += 1
    return env_history, False


def adjust_trial_number(trial_log_path: str, is_fail: bool = False) -> str:
    pattern = r"trial_(\d+)\.log"

    def replace(match):
        return "trial_fail.json" if is_fail else "trial_inf.json"

    result = re.sub(pattern, replace, trial_log_path)
    return result


def adjust_trial_number2(trial_log_path: str) -> str:
    pattern = r"trial_(\d+)\.log"

    def replace(match):
        return "trial_cache.json"

    result = re.sub(pattern, replace, trial_log_path)
    return result


def get_offline_embedding(des):
    with open("embedding.json", "r", encoding="utf-8") as file:
        embdata = json.load(file)
    for a, b in embdata:
        if des == a:
            return b
    else:
        return []


def enumerate_splits(string):
    lines = [line for line in string.split("\n") if line]
    result = []

    for i in range(len(lines) - 1):
        first_part = "\n".join(lines[: i + 1])
        second_part = "\n".join(lines[i + 1 :])
        result.append((first_part, second_part))

    return result


def replace_lines(text, prefix, suffix):
    replaced_text = re.sub(
        r"^>(.*)$", lambda m: f"{prefix}{m.group(1)}{suffix}", text, flags=re.MULTILINE
    )
    return replaced_text


def print_first_n_lines(text, n=5):
    lines = text.split('\n')
    ret=""
    for line in lines[:n]:
        ret+=line+"\n"
    return ret

def run_trial(
    cluster_size,
    cluster_counter,
    model_server: ModelServer,
    plan_model_size: str,
    trial_log_path: str,
    world_log_path: str,
    trial_idx: int,
    env_configs: List[Dict[str, Any]],
    mem_size: int = 0,
    specific_cluster_name: str = None,
    mem_selection_method: str = "fifo",
    use_success_trajectory: bool = False,
    trajactory_search_method: str = "knn",
    in_context_trajactory_size: int = 3,
    online_embedding_model_size: str = "2",
) -> List[Dict[str, Any]]:
    importlib.reload(alfworld)
    importlib.reload(alfworld.agents.environment)
    success_in_each_cluster = {
        "clean": 0,
        "put": 0,
        "cool": 0,
        "puttwo": 0,
        "examine": 0,
        "heat": 0,
    }
    with open("base_config.yaml") as reader:
        config = yaml.safe_load(reader)
    split = "eval_out_of_distribution"

    assert (
        not use_success_trajectory
        and trajactory_search_method is None
        and in_context_trajactory_size == 0
    ) or (
        use_success_trajectory
        and trajactory_search_method in ["knn", "random"]
        and in_context_trajactory_size > 0
    ), "If use_success_trajectory is False, then in_context_trajactory_size should be 0."

    env = getattr(alfworld.agents.environment, config["env"]["type"])(
        config, train_eval=split
    )
    env = env.init_env(batch_size=1)

    num_successes: int = 0
    num_success_increase: int = 0
    actcnt = 0
    last_trial_log_path: str = adjust_trial_number(trial_log_path)
    fail_db_path = adjust_trial_number(trial_log_path, is_fail=True)
    emb_cache_path = adjust_trial_number2(trial_log_path)
    success_data = {
        "clean": [],
        "put": [],
        "cool": [],
        "puttwo": [],
        "examine": [],
        "heat": [],
    }

    increase_success = {
        "clean": [],
        "put": [],
        "cool": [],
        "puttwo": [],
        "examine": [],
        "heat": [],
    }

    if not os.path.exists(last_trial_log_path):
        with open(last_trial_log_path, "w", encoding="utf-8") as file:
            print("build success log")
            json.dump(
                {
                    "clean": [],
                    "put": [],
                    "cool": [],
                    "puttwo": [],
                    "examine": [],
                    "heat": [],
                },
                file,
                indent=4,
            )

    with open(last_trial_log_path, "r", encoding="utf-8") as file:
        success_data = json.load(file)

    fail_data = {}
    emb_cache = {}

    if not os.path.exists(emb_cache_path):
        with open(emb_cache_path, "w", encoding="utf-8") as file:
            print("build embed cache")
            json.dump({}, file)

    with open(emb_cache_path, "r", encoding="utf-8") as file:
        emb_cache = json.load(file)

    if not os.path.exists(fail_db_path):
        with open(fail_db_path, "w", encoding="utf-8") as file:
            print("build fail log")
            json.dump({}, file)

    with open(fail_db_path, "r", encoding="utf-8") as file:
        fail_data = json.load(file)

    trajectories = []
    embedding_array = np.zeros((0, EMBEDDING_DIM[online_embedding_model_size]))
    huge_trajectories = []
    huge_ary = np.zeros((0, EMBEDDING_DIM[online_embedding_model_size]))
    if trajactory_search_method == "knn":
        for key in success_data:
            for des, trj in success_data[key]:
                # vec = get_offline_embedding(des)
                trj_cut = (trj + "\n").split("Here is the task:")[-1].strip()
                if trj_cut in emb_cache:
                    vec = emb_cache[trj_cut]
                else:
                    vec = model_server.get_completion_or_embedding(
                        online_embedding_model_size,
                        message=trj_cut,
                        get_embedding=True,
                    )
                    emb_cache[trj_cut] = vec
                trajectories.append(trj)
                embedding_array = np.vstack((embedding_array, np.array(vec)))
    elif trajactory_search_method == "random":
        for key in success_data:
            for des, trj in success_data[key]:
                trajectories.append(trj)

    emb_db = faiss.IndexFlatL2(EMBEDDING_DIM[online_embedding_model_size])
    emb_db.add(embedding_array.astype("float32"))
    huge_db = faiss.IndexFlatL2(EMBEDDING_DIM[online_embedding_model_size])
    huge_db.add(huge_ary.astype("float32"))
    final_fail_db = {}

    for z, env_config in enumerate(env_configs):

        ob, info = env.reset()
        ob = "\n".join(ob[0].split("\n\n")[1:])
        env_description = ob.strip()
        env_vec = (
            np.array(get_offline_embedding(env_description))
            .reshape(1, -1)
            .astype("float32")
        )
        name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])

        print(f"using {name}")

        if env_config["is_success"]:
            for i, (k, v) in enumerate(PREFIXES.items()):
                if name.startswith(k):
                    success_in_each_cluster[v] += 1
            num_successes += 1
            actcnt += 1
            with open(world_log_path, "a", encoding="utf-8") as wf:
                wf.write(
                    f"""

==============================================================
Environment #{z}
Trial #{trial_idx}
Game: {name}
SUCCESS
==============================================================

"""
                )
            with open(trial_log_path, "a", encoding="utf-8") as wf:
                wf.write(
                    f"""

==============================================================
Environment #{z}
Trial #{trial_idx}
Game: {name}
SUCCESS
==============================================================

"""
                )
            continue

        for i, (k, v) in enumerate(PREFIXES.items()):
            if name.startswith(k):
                env_configs[z]["cluster"] = v
                if v in cluster_counter:
                    if len(cluster_counter[v]) < cluster_size and (
                        specific_cluster_name is None or v == specific_cluster_name
                    ):
                        cluster_counter[v].append(name)
                else:
                    if specific_cluster_name is None or v == specific_cluster_name:
                        cluster_counter[v] = [name]
                    else:
                        cluster_counter[v] = []
                base_prompt = (
                    "Interact with a household to solve a task. Here are a few examples.\n"
                    + d[f"react_{v}_1"]
                    + d[f"react_{v}_0"]
                    + "\n"
                )
                if (
                    trajectories
                    and use_success_trajectory
                    and in_context_trajactory_size > 0
                ):
                    assert trajactory_search_method in ["knn", "random"]
                    if trajactory_search_method == "knn":
                        if name in fail_data:
                            fail_key=fail_data[name].strip()
                        else:
                            fail_key=env_description.strip()
                        fail_vec = model_server.get_completion_or_embedding(
                            online_embedding_model_size,
                            message=fail_key,
                            get_embedding=True,
                        )
                        indices = [[]]
                        dist=[]
                        for index, row in np.ndenumerate(embedding_array):
                            if index[1] == 0:
                                realN = (1.0 / np.linalg.norm(np.array(fail_vec) - embedding_array[index[0]]))
                                dist.append((realN,index[0]))
                        dist.sort(key=lambda x: x[0], reverse=True)
                        sz_now=min(len(trajectories),in_context_trajactory_size)
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
                            indices[0].append(y)
                    elif trajactory_search_method == "random":
                        indices[0] = random.sample(
                            range(len(trajectories)), min(len(trajectories),in_context_trajactory_size)
                        )
                    for i in indices[0]:
                        base_prompt += (
                            (trajectories[i] + "\n")
                            .split("Here is the task:")[-1]
                            .strip()
                        ) + "\n"
                base_prompt = replace_lines(
                    base_prompt,
                    "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n",
                    "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n",
                )
                base_prompt += "\n\nExamples finished.\n\n"
                if env_config["principle"]:
                    base_prompt += "Here are the core principles you should follow as much as possible in your planning:\n[Principles start]\n"
                    for pi in env_config["principle"]:
                        base_prompt += pi + "\n"
                    base_prompt += "\n[Principles end]\n\n"
                inclus = False
                final_env_history = ""
                is_success = False
                if name in cluster_counter[v]:
                    actcnt += 1
                    inclus = True
                    final_env_history, is_success = alfworld_run(
                        model_server,
                        plan_model_size,
                        env,
                        base_prompt,
                        env_config["memory"],
                        to_print=True,
                        ob=ob,
                        mem_size=mem_size,
                        mem_selection_method=mem_selection_method,
                    )
                else:
                    env_configs[z]["skip"] = True

                # update env config
                if is_success:
                    status_str: str = f"""

==============================================================
Environment #{z}
Trial #{trial_idx}
Game: {name}
SUCCESS
==============================================================

"""
                    env_configs[z]["is_success"] = True
                    num_successes += 1
                    num_success_increase += 1
                    success_in_each_cluster[v] += 1
                elif inclus:
                    status_str: str = f"""

==============================================================
Environment #{z}
Trial #{trial_idx}
Game: {name}
FAIL
==============================================================

"""
                else:
                    status_str: str = f"""
                    
==============================================================
Environment #{z}
Trial #{trial_idx}
SKIP
==============================================================

"""
                # log to world log
                assert world_log_path.endswith(".log")
                if not os.path.exists(world_log_path):
                    os.system(f"touch {world_log_path}")
                with open(world_log_path, "a", encoding="utf-8") as f:
                    f.write(status_str + "\n")

                if is_success:
                    increase_success[v].append(
                        (env_description, str(final_env_history))
                    )
                    trj_cut = (str(final_env_history) + "\n").split("Here is the task:")[-1].strip()
                    vec = model_server.get_completion_or_embedding(
                        online_embedding_model_size,
                        message=trj_cut,
                        get_embedding=True,
                    )
                    emb_cache[trj_cut] = vec
                    trajectories.append(str(final_env_history))
                    embedding_array = np.vstack((embedding_array, np.array(vec)))
                else:
                    final_fail_db[name] = (
                        (str(final_env_history) + "\n")
                        .split("Here is the task:")[-1]
                        .strip()
                    )

                with open(trial_log_path, "a", encoding="utf-8") as wf:
                    wf.write(
                        f"""

==============================================================
Environment #{z}
{str(final_env_history)}
STATUS: {"OK" if is_success else "FAIL"}
==============================================================

"""
                    )
    env.close()

    log_str: str = f"""

==============================================================
SUCCESS: {num_successes}
INCREASE SUCCESS: {num_success_increase}
FAIL: {actcnt - num_successes}
TOTAL: {actcnt}
ACCURACY: {round(num_successes / actcnt, 2)}
CLUSTER SUCCEESS: {str(success_in_each_cluster)}
==============================================================

"""

    with open(last_trial_log_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                key: success_data.get(key, []) + increase_success.get(key, [])
                for key in success_data.keys() | increase_success.keys()
            },
            file,
            indent=4,
        )
    with open(fail_db_path, "w", encoding="utf-8") as file:
        json.dump(
            final_fail_db,
            file,
            indent=4,
        )

    with open(emb_cache_path, "w", encoding="utf-8") as file:
        json.dump(
            emb_cache,
            file,
            indent=4,
        )

    with open(trial_log_path, "a", encoding="utf-8") as wf:
        wf.write(log_str)
    with open(world_log_path, "a", encoding="utf-8") as wf:
        wf.write(log_str + "\n")
    return env_configs, cluster_counter
