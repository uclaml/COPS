import os
from IPython import embed

os.environ["PYTHONUTF8"] = "1"
import json
import argparse
from utils import ModelServer

from alfworld_trial import run_trial

from generate_reflections import update_memory
from typing import Any, List, Dict
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, help="The number of trials to run")
    parser.add_argument(
        "--num_envs", type=int, help="The number of environments per trial"
    )
    parser.add_argument(
        "--mem_size",
        type=int,
        help="The size of the memory that will be used in the memory. -1 means unlimited mem, 0 means 0 mem.",
        default=0,
    )
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--is_resume", action="store_true", help="To resume run")
    parser.add_argument(
        "--resume_dir", type=str, help="If resume, the logging directory", default=""
    )
    parser.add_argument(
        "--start_trial_num",
        type=int,
        help="The start trial num, if not resume, should be 0.",
        default=0,
    )
    parser.add_argument(
        "--plan_model_size", type=str, help="only support 8, 70", default="70"
    )
    parser.add_argument(
        "--reflect_model_size", type=str, help="only support 8, 70", default="70"
    )
    parser.add_argument(
        "--online_embedding_model_size", type=str, help="only support 2, 7", default="2"
    )
    parser.add_argument(
        "--cluster_size",
        type=int,
        help="How many instances to run in each cluster",
        default=5,
    )
    parser.add_argument(
        "--specific_cluster_name",
        type=str,
        default=None,
        help="Whether to only run on a specific cluster of envs",
    )
    parser.add_argument(
        "--mem_selection_method",
        type=str,
        default="fifo",
        help="The method to select memory for planning, only fifo/fix/random.",
    )
    parser.add_argument(
        "--log_file_path",
        type=str,
        help="The path to the log file",
        default="",
        required=True,
    )
    parser.add_argument(
        "--use_success_trajectory",
        action="store_true",
        help="Whether to use success trajectories in planning base prompt.",
        default=False,
    )

    parser.add_argument(
        "--trajactory_search_method",
        type=str,
        default=None,
        help="The method to select memory for planning, only knn/random.",
    )
    parser.add_argument(
        "--in_context_trajactory_size",
        type=int,
        default=3,
        help="The size of in context trajactory size.",
    )
    args = parser.parse_args()

    assert args.log_file_path != "", "Log file path should be provided"
    assert args.num_trials > 0, "Number of trials should be positive"
    assert args.num_envs > 0, "Number of environments should be positive"
    assert (
        args.resume_dir == args.run_name or args.resume_dir == ""
    ), "Should resume from previous directory"
    assert args.mem_selection_method in [
        "fifo",
        "fix",
        "random",
    ], "Invalid memory selection method"
    assert args.specific_cluster_name in [
        None,
        "clean",
        "put",
        "heat",
        "cool",
        "examine",
        "puttwo",
    ], "Invalid cluster name used"
    assert (
        not args.use_success_trajectory
        and args.trajactory_search_method is None
        and args.in_context_trajactory_size == 0
    ) or (
        args.use_success_trajectory
        and args.trajactory_search_method in ["knn", "random"]
        and args.in_context_trajactory_size > 0
    ), "If use_success_trajectory is False, then in_context_trajactory_size should be 0."
    assert args.trajactory_search_method in [
        "knn",
        "random",
        None,
    ], "Invalid search method"
    assert args.plan_model_size in ["8", "70"], "Invalid plan model size"
    assert args.reflect_model_size in ["8", "70"], "Invalid reflect model size"
    assert args.online_embedding_model_size in [
        "2",
        "7",
    ], "Invalid online embedding size"
    return args


def main(args) -> None:
    if args.is_resume:
        if not os.path.exists(args.resume_dir):
            raise ValueError(f"Resume directory `{args.resume_dir}` does not exist")
        logging_dir = args.resume_dir

        # load previous environment configs
        env_config_path: str = os.path.join(
            args.resume_dir, f"env_results_trial_{args.start_trial_num - 1}.json"
        )
        if not os.path.exists(env_config_path):
            raise ValueError(
                f"Environment config file `{env_config_path}` does not exist"
            )
        with open(env_config_path, "r", encoding="utf-8") as rf:
            env_configs: List[Dict[str, Any]] = json.load(rf)
    else:
        # Create new run directory
        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)
        logging_dir = args.run_name

        # initialize environment configs
        env_configs: List[Dict[str, Any]] = []

        env_configs = [
            {
                "name": f"env_{i}",
                "memory": [],
                "principle": [],
                "is_success": False,
                "skip": False,
                "cluster": "",
            }
            for i in range(args.num_envs)
        ]
    config_path = os.path.join(logging_dir, "config.json")
    model_server = ModelServer(config_path=config_path)
    world_log_path: str = os.path.join(logging_dir, "world.log")
    os.system(f"touch {world_log_path}")
    with open(world_log_path, "a", encoding="utf-8") as wf:
        if args.is_resume:
            info_string = f"""
                        
==============================================================
RESUME
Run name: {args.run_name}
Number of trials: {args.num_trials}
Number of max environments: {args.num_envs}
Number of instances per cluster: {args.cluster_size}
Resume trial number: {args.start_trial_num}
Sending all logs to: {args.run_name}
Setting memory size to: {args.mem_size}
Plan model size: {args.plan_model_size}
Reflect model size: {args.reflect_model_size}
Online embedding size: {args.online_embedding_model_size}
Specific cluster name: {args.specific_cluster_name}
Log file path: {args.log_file_path}
Use success trajectory: {args.use_success_trajectory}
Trajactory search method: {args.trajactory_search_method}
In context trajactory size: {args.in_context_trajactory_size}
==============================================================

    """
            wf.write(info_string)
        else:
            info_string = f"""
                        
==============================================================
START
Run name: {args.run_name}
Number of trials: {args.num_trials}
Number of max environments: {args.num_envs}
Number of instances per cluster: {args.cluster_size}
Start trial number: {args.start_trial_num}
Sending all logs to: {args.run_name}
Setting memory size to: {args.mem_size}
Plan model size: {args.plan_model_size}
Reflect model size: {args.reflect_model_size}
Online embedding size: {args.online_embedding_model_size}
Specific cluster name: {args.specific_cluster_name}
Log file path: {args.log_file_path}
Use success trajectory: {args.use_success_trajectory}
Trajactory search method: {args.trajactory_search_method}
In context trajactory size: {args.in_context_trajactory_size}
==============================================================

    """
            wf.write(info_string)

    with open(config_path, "w", encoding="utf-8") as wf:
        info_dict = vars(args)
        info_dict["is_running"] = True
        json.dump(info_dict, wf, indent=4)

    trial_idx = args.start_trial_num
    cluster_counter = {}

    while trial_idx < args.num_trials:
        with open(world_log_path, "a", encoding="utf-8") as wf:
            wf.write(
                f"""
                     
==============================================================
Start Trial #{trial_idx}
==============================================================

"""
            )

        # set paths to log files
        trial_log_path: str = os.path.join(args.run_name, f"trial_{trial_idx}.log")
        os.system(f"touch {trial_log_path}")
        trial_env_configs_log_path: str = os.path.join(
            args.run_name, f"env_results_trial_{trial_idx}.json"
        )
        if os.path.exists(trial_log_path):
            open(trial_log_path, "w").close()
        if os.path.exists(trial_env_configs_log_path):
            open(trial_env_configs_log_path, "w").close()

        env_configs, cluster_counter = run_trial(
            args.cluster_size,
            cluster_counter,
            model_server,
            args.plan_model_size,
            trial_log_path,
            world_log_path,
            trial_idx,
            env_configs,
            args.mem_size,
            specific_cluster_name=args.specific_cluster_name,
            mem_selection_method=args.mem_selection_method,
            use_success_trajectory=args.use_success_trajectory,
            trajactory_search_method=args.trajactory_search_method,
            in_context_trajactory_size=args.in_context_trajactory_size,
            online_embedding_model_size=args.online_embedding_model_size,
        )


        if args.mem_size != 0:

            env_configs: List[Dict[str, Any]] = update_memory(
                model_server,
                args.reflect_model_size,
                trial_log_path,
                env_configs,
                mem_size=args.mem_size,
            )

        # log env configs for trial
        with open(trial_env_configs_log_path, "w", encoding="utf-8") as wf:
            json.dump(env_configs, wf, indent=4)

        # log world for trial
        with open(world_log_path, "a", encoding="utf-8") as wf:
            wf.write(
                f"""
                     
==============================================================
End Trial #{trial_idx}
==============================================================

"""
            )
        trial_idx += 1


if __name__ == "__main__":
    args = get_args()
    main(args)
