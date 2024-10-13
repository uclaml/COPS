import openai
from utils import ModelServer
import time
import re

server = None


def build_server(config_path):
    global server
    server = ModelServer(config_path=config_path)


def split_and_keep_prefixes(s, delimiters):
    regex_pattern = f"({'|'.join(map(re.escape, delimiters))})"
    parts = re.split(regex_pattern, s)
    result = [parts[0]]
    for i in range(1, len(parts), 2):
        result.append(parts[i] + (parts[i + 1] if i + 1 < len(parts) else ""))
    return result


def online_embed(traj):
    return server.get_completion_or_embedding(
        "7",
        message=traj,
        get_embedding=True,
    )

ret_in=0
ret_out=0
def gpt(prompt, model_size="8", temperature=1.0, max_tokens=100, n=1) -> list:
    def call_openai_api(messages, model_size, temperature, max_tokens, n):
        global ret_in
        global ret_out
        outputs = []
        while n > 0:
            cnt = min(n, 20)
            n -= cnt
            res = server.get_completion_or_embedding(
                model_size, messages, temperature, max_tokens
            )
            ret_out+=res.usage.completion_tokens
            ret_in+=res.usage.prompt_tokens
            outputs.extend(
                [
                    re.sub(r"^Action:", "", choice.message.content)
                    for choice in res.choices
                ]
            )
        return outputs,res.usage.prompt_tokens,res.usage.completion_tokens

    messages = []
    parts = re.split(r"(Action:|Observation:|Instruction:)", prompt)

    result = [parts[0].strip()]
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            result.append(parts[i] + " " + parts[i + 1].strip())
    result.pop()
    last_obs=""
    for msg in result:
        if msg.startswith("Action"):
            messages.append({"role": "assistant", "content": msg})
        if msg.startswith("Observation"):
            messages.append({"role": "user", "content": msg})
            last_obs=msg
        if msg.startswith("Instruction"):
            messages.append({"role": "user", "content": msg})
    return call_openai_api(
        messages, model_size=model_size, temperature=temperature, max_tokens=max_tokens, n=n
    )
