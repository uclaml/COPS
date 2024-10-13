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

tot_in=0
tot_out=0

class ModelServer:
    
    def get_completion_or_embedding(
        self,
        model_size: str,
        message,
        temperature: float = 0.0,
        max_tokens: int = 256,
        get_embedding: bool = False,
    ) -> str:
        global tot_in
        global tot_out
        assert model_size in ["70", "8", "7"]
       
        if not get_embedding:
            assert type(message) == list, "Message should be a list."
            response = client.chat.completions.create(
                model=model_name,
                messages=message,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|eot_id|>", "\nObservation", "Observation"],
            )
            tot_out+=response.usage.completion_tokens
            tot_in+=response.usage.prompt_tokens
        else:
            assert type(message) == str, "Message should be a string."
            response = client.embeddings.create(
                model=model_name,
                input=message,
            )
        

        if get_embedding:
            return response.data[0].embedding
        else:
            return response


