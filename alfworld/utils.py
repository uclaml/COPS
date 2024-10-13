import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["PYTHONUTF8"] = "1"
import time
from typing import Dict, List
import openai
import random, json
from IPython import embed
import random
import math


class EnvironmentHistory:
    def __init__(
        self,
        base_query: str,
        start_info,
        memory: List[str],
        history: List[Dict[str, str]] = [],
    ) -> None:

        def _get_base_query(base_query: str, start_info: str, memory: List[str]) -> str:
            query = base_query

            # add memory if it exists

            query += f"\nHere is the task:\n{start_info}"
            if len(memory) > 0:
                query += "\n\nBelow are your reflection memory for the task, you should apply them wisely in your planning:\n[memory start]\n"
                for i, m in enumerate(memory):
                    query += f"\nReflection from Trial {i}:\n{m.strip()}"
                query += "\n[memory end]\n"
            return query

        self._cur_query: str = f"{_get_base_query(base_query, start_info, memory)}"
        self._history: List[Dict[str, str]] = history
        self._last_action: str = ""
        self._is_exhausted: bool = False

    def add(self, label: str, value: str) -> None:
        assert label in ["action", "observation", "human_edit"]
        self._history += [
            {
                "label": label,
                "value": value,
            }
        ]
        if label == "action":
            if value == self._last_action:
                self._is_exhausted = True
            else:
                self._last_action = value

    def check_is_exhausted(self) -> bool:
        return self._is_exhausted

    def reset(self) -> None:
        self._history = []

    def __str__(self) -> str:
        s: str = self._cur_query + "\n"
        for i, item in enumerate(self._history):
            if item["label"] == "action":
                s += f'> {item["value"]}'
            elif item["label"] == "observation":
                s += item["value"]
            # NOT CURRENTLY SUPPORTED
            elif item["label"] == "human_edit":
                s += f'[human edit]: {item["value"]}'
            if i != len(self._history) - 1:
                s += "\n"
        return s


class ModelServer:
    
    def get_completion_or_embedding(
        self,
        model_size: str,
        message,
        temperature: float = 0.0,
        max_tokens: int = 256,
        get_embedding: bool = False,
    ) -> str:
        assert model_size in ["70", "8", "7"]
        
        if not get_embedding:
            assert type(message) == list, "Message should be a list."
            response = client.chat.completions.create(
                model=model_name,
                messages=message,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|eot_id|>"],
            )
        else:
            assert type(message) == str, "Message should be a string."
            response = client.embeddings.create(
                model=model_name,
                input=message,
            )
        return (
            str(response.choices[0].message.content)
            if not get_embedding
            else response.data[0].embedding
        )

