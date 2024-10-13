import os

os.environ["PYTHONUTF8"] = "1"
import random
from utils import ModelServer

from typing import List, Dict, Any

with open("./reflexion_few_shot_examples.txt", "r", encoding="utf-8") as f:
    FEW_SHOT_EXAMPLES = f.read()


def _get_scenario(s: str) -> str:
    """Parses the relevant scenario from the experience log."""
    return s.split("Here is the task:")[-1].strip()


def _generate_reflection_query(log_str: str, memory: List[str]) -> str:
    """Allows the Agent to reflect upon a past experience."""
    scenario: str = _get_scenario(log_str)
    query: str = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Remember that your plan must be very concise and your entire output must only limited to a single line of words. Here are two examples:

{FEW_SHOT_EXAMPLES}

Here is your failed experience:
{scenario}"""

    query += "\n\nNew plan:"
    return query


def update_memory(
    model_server: ModelServer,
    reflection_model_size: str,
    trial_log_path: str,
    env_configs: List[Dict[str, Any]],
    mem_size: int = -1,
) -> List[Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    with open(trial_log_path, "r", encoding="utf-8") as f:
        full_log: str = f.read()

    env_logs = full_log.split(
        "==============================================================\n\n\n\n=============================================================="
    )

    faile = []
    env_logs = env_logs[:-1]
    env_logs[0] = env_logs[0].strip(
        "\n\n==============================================================\n"
    )
    env_logs = [each.strip() for each in env_logs]
    assert all(
        ["Environment #" in env_logs[i] for i in range(len(env_logs))]
    ), "Parsing error"
    #! 不要修改 print 的这串字符串，因为这串字符串是用来分割日志的
    #! env_logs[0] 需要特殊处理，而最原始的 env_logs[-1] 是胜率总结
    for i, env in enumerate(env_configs):

        # if unsolved, get reflection and update env config
        if not env["is_success"] and not env["skip"]:
            faile.append(i)
            assert mem_size != 0, "Memory size should not be 0."
            #! Reflection 的 mem selection 都是 FIFO
            if mem_size != -1 and len(env["memory"]) >= mem_size:
                memory: List[str] = env["memory"][-mem_size:]
            else:
                memory: List[str] = env["memory"]
            reflection_query: str = _generate_reflection_query(
                env_logs[i].strip(), memory
            )
            print("start reflection env:")
            print(i)
            messages = [{"role": "user", "content": reflection_query}]
            reflection: str = model_server.get_completion_or_embedding(reflection_model_size, messages)  # type: ignore
            env_configs[i]["memory"] += [reflection]

    return env_configs  #! 删去 general reflection 的逻辑
