import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import itertools
import time
import json
import sys

from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm

import logging
logging.basicConfig(level = logging.INFO)

import numpy as np
import torch
from transformers import AutoTokenizer
import re

import torch
import torch._dynamo.config
import torch._inductor.config

from phi_model import Transformer


def get_task(sentence):
    task = ""
    quote_type = ""
    start_pos = -1
    end_pos = -1
    # Remove multiple spaces.
    while True:
        if sentence.find('  ') >= 0:
            sentence = sentence.replace('  ', ' ')
        else:
            break
    
    # Reponse start patterns.
    task_start_patterns = ["t': '", "t':'", "t' : '", "t' :'", 't": "', 't":"', 't" : "', 't" :"', "'t' '", '"t" "']
    
    for p, start_pattern in enumerate(task_start_patterns):
        if start_pattern in sentence:
            # quote_type = 
            start_pos = sentence.find(start_pattern)
            if start_pos >= 0:

                # Get quote type.
                quote_type = "'"
                if '"' in start_pattern:
                    quote_type = '"'

                # Find end of task.
                end_pos = sentence.find(quote_type, start_pos + len(start_pattern))
                if end_pos >= 0:
                    task = sentence[start_pos + len(start_pattern):end_pos]
                else:
                    task = sentence[start_pos + len(start_pattern):]

    return (task.strip(), quote_type, start_pos, end_pos)

def get_response(sentence):
    response = ""
    quote_type = ""
    start_pos = -1
    end_pos = -1
    # Remove multiple spaces.
    while True:
        if sentence.find('  ') >= 0:
            sentence = sentence.replace('  ', ' ')
        else:
            break
    
    # Reponse start patterns.
    response_start_patterns = ["r': '", "r':'", "r' : '", "r' :'", 'r": "', 'r":"', 'r" : "', 'r" :"', "'r' '", '"r" "']

    for p, start_pattern in enumerate(response_start_patterns):
        if start_pattern in sentence:
            # quote_type = 
            start_pos = sentence.find(start_pattern)
            if start_pos >= 0:

                # Get quote type.
                quote_type = "'"
                if '"' in start_pattern:
                    quote_type = '"'

                # Find end of response.
                end_pos = sentence.find(quote_type, start_pos + len(start_pattern))
                if end_pos >= 0:
                    response = sentence[start_pos + len(start_pattern):end_pos]
                else:
                    response = sentence[start_pos + len(start_pattern):]

    return (response.strip(), quote_type, start_pos, end_pos)

def get_mutation(sentence):
    mutation = ""
    quote_type = ""
    start_pos = -1
    end_pos = -1
    # Remove multiple spaces.
    while True:
        if sentence.find('  ') >= 0:
            sentence = sentence.replace('  ', ' ')
        else:
            break
    
    # Reponse start patterns.
    mutation_start_patterns = ["m': '", "m':'", "m' : '", "m' :'", 'm": "', 'm":"', 'm" : "', 'm" :"', "'m' '", '"m" "']

    for p, start_pattern in enumerate(mutation_start_patterns):
        if start_pattern in sentence:
            # quote_type = 
            start_pos = sentence.find(start_pattern)
            if start_pos >= 0:

                # Get quote type.
                quote_type = "'"
                if '"' in start_pattern:
                    quote_type = '"'

                # Find end of mutation.
                end_pos = sentence.find(quote_type, start_pos + len(start_pattern))
                if end_pos >= 0:
                    mutation = sentence[start_pos + len(start_pattern):end_pos]
                else:
                    mutation = sentence[start_pos + len(start_pattern):]

    return (mutation.strip(), quote_type, start_pos, end_pos)

def get_llm_output_json(json_string):
    # Remove the outer curly braces and split by ':'
    llm_json_output = {}
    try:
        llm_json_output = json.loads(json_string)
    except Exception as e:
        print("No valid json output ... trying to fix it: ", e)
        llm_json_output = {
            "r": "Please be more specific. What can I get for you today?", 
            "m": "", 
            "t":"greeting%default;"
        }

        response = get_response(json_string)
        if response[2] >= 0 and response[3] >= 0:
            llm_json_output["r"] = response[0]

        mutation = get_mutation(json_string)
        if mutation[2] >= 0 and mutation[3] >= 0:
            llm_json_output["m"] = mutation[0]

        task = get_task(json_string)
        if task[2] >= 0 and task[3] >= 0:
            llm_json_output["t"] = task[0]
    return json.dumps(llm_json_output)

def replace_num_total(test_str, total_price):
    number = ""
    if "\u20ac" in test_str:
        start_pos = test_str.find("\u20ac")
        end_pos = test_str.find(" ", start_pos)
        if end_pos > 0:
            number = test_str[start_pos:end_pos]
        else:
            number = test_str[start_pos:]

    if "€" in test_str:
        start_pos = test_str.find("€")
        end_pos = test_str.find(" ", start_pos)
        if end_pos > 0:
            number = test_str[start_pos:end_pos]
        else:
            number = test_str[start_pos:]

    endings = [".", ",", "!", "?"]
    for ending in endings:
        if number.endswith(ending):
            number = number[:-1]

    if len(number) > 0:
        test_str = test_str.replace(number, total_price)
    else:
        test_str = test_str.replace("[[TOTAL]]", total_price)


    if "\u20ac" in test_str:
        test_str = test_str.replace("\u20ac", "")
    if "€" in test_str:
        test_str = test_str.replace("\u20ac", "")

    return test_str


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif "cpu" in device:
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future


def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, **sampling_kwargs):
    new_tokens, new_probs = [], []

    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)

    return new_tokens, new_probs



def encode_tokens(tokenizer, chat, bos=False, device='cuda'):
    tokenized_sample = tokenizer(
            chat, add_special_tokens=False, return_tensors="pt"
        ).to(device)
    tokens = tokenized_sample["input_ids"][0]
    return tokens.to(torch.int)


class PhiEngine:
    def __init__(self):
        pass
    
    def _load_model(self, checkpoint_path, device, precision):
        with torch.device('meta'):
            self.model = Transformer.from_name("phi-2")

        if "int8" in str(checkpoint_path):
            print("Using int8 weight-only quantization!")
            from quantize import WeightOnlyInt8QuantHandler
            simple_quantizer = WeightOnlyInt8QuantHandler(self.model)
            self.model = simple_quantizer.convert_for_runtime()

        if "int4" in str(checkpoint_path):
            print("Using int4 quantization!")
            path_comps = checkpoint_path.name.split(".")
            assert path_comps[-2].startswith("g")
            groupsize = int(path_comps[-2][1:])
            from quantize import WeightOnlyInt4QuantHandler
            simple_quantizer = WeightOnlyInt4QuantHandler(self.model, groupsize)
            self.model = simple_quantizer.convert_for_runtime()
        
        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        self.model.load_state_dict(checkpoint, assign=True)
        self.model.to(device=device, dtype=precision)
    
    def initialize(self, checkpoint_path, device, max_new_tokens):
        with open("assets/phi-2/col-data-v6.json", "r") as file:
            data = json.load(file)
        self.system_prompt = data[-1]["prompt"]
        precision = torch.bfloat16
        logging.info("[LLM INFO:] Loading model ...")
        t0 = time.time()
        self._load_model(checkpoint_path, device, precision)
        logging.info(f"[LLM INFO:] Time to load model: {time.time() - t0:.02f} seconds")

        device_sync(device=device) # MKG
        
        self.tokenizer = AutoTokenizer.from_pretrained("assets/phi-2/")

        t0 = time.time()
        if compile:
            global decode_one_token
            decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True, dynamic=True)
            

        torch.manual_seed(1234)
        self.model_size = sum([p.numel() * p.dtype.itemsize for p in itertools.chain(self.model.parameters(), self.model.buffers())])
        input_text = self.format_prompt_chatml("Hello, I am Ready.", [], system_prompt=self.system_prompt)
        encoded = encode_tokens(self.tokenizer, input_text, bos=False, device=device)
        logging.info("[LLM INFO:] Phi torch compile warm up")
        for i in tqdm(range(5), desc="warming up Phi torch compile model"):
            y = self.generate(
                    encoded,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    top_k=200,
                )
            device_sync(device=device) # MKG
        logging.info(f"[LLM INFO:] Compilation time: {time.time() - t0:.02f} seconds")
        self.last_prompt = None
        self.last_output = None
        
    def format_prompt_qa(self, prompt, conversation_history):
        formatted_prompt = ""
        for user_prompt, llm_response in conversation_history:
            formatted_prompt += f"Instruct: {user_prompt}\nOutput:{llm_response}\n"
        return f"{formatted_prompt}Instruct: {prompt}\nOutput:"
    
    def format_prompt_chat(self, prompt, conversation_history):
        formatted_prompt = ""
        for user_prompt, llm_response in conversation_history:
            formatted_prompt += f"Alice: {user_prompt}\nBob:{llm_response}\n"
        return f"{formatted_prompt}Alice: {prompt}\nBob:"

    def format_prompt_chatml(self, prompt, conversation_history, system_prompt=""):
        messages = [{"role": "system", "content": system_prompt}]
        for user_prompt, llm_response in conversation_history:
            messages.append({"role": "user", "content": user_prompt})
            messages.append({"role": "assistant", "content": llm_response})
        messages.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(messages, tokenize=False).strip() 

    
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        **sampling_kwargs
    ) -> torch.Tensor:
        """
        Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
        """

        # create an empty tensor of the expected final shape and fill in the current tokens
        T = prompt.size(0)
        T_new = T + max_new_tokens
        max_seq_length = min(T_new, self.model.config.block_size - 1)
        # max_length = self.model.config.block_size
        logging.info("=============================================================")
        logging.info(f"[LLM INFO:] max_seq_length: {self.model.config.block_size - 1}")
        logging.info("=============================================================")
        device, dtype = prompt.device, prompt.dtype
        with torch.device(device):
            print(device)
            self.model.setup_caches(max_batch_size=1, max_seq_length=self.model.config.block_size - 1)
            
        # create an empty tensor of the expected final shape and fill in the current tokens
        empty = torch.empty(self.model.config.block_size, dtype=dtype, device=device)
        empty[:T] = prompt
        seq = empty
        input_pos = torch.arange(0, T, device=device)

        next_token = prefill(self.model, prompt.view(1, -1), input_pos, **sampling_kwargs)

        seq[T] = next_token

        input_pos = torch.tensor([T], device=device, dtype=torch.int)
        
        generated_tokens, _ = decode_n_tokens(self.model, next_token.view(1, -1), input_pos, self.model.config.block_size - T - 1, **sampling_kwargs)
        seq[T + 1:] = torch.cat(generated_tokens)

        return seq
    
    def run(
        self,
        transcription_queue=None,
        llm_queue=None,
        audio_queue=None,
        prompt = "Hello, my name is Samantha. I am a doctor.",
        checkpoint_path = Path("assets/phi-2/model.pth"),
        compile = True,
        temperature = 0.8,
        top_k = 200,
        max_new_tokens = 240,
        device='cuda'
    ):
        self.initialize(checkpoint_path, device, max_new_tokens)
        conversation_history = {}
        while True:
            # Get the last transcription output from the queue
            # logging.info(f"[LLM INFO]: {transcription_queue.qsize()}")
            transcription_output = transcription_queue.get()
            if transcription_queue.qsize() != 0:
                continue
            
            if transcription_output["uid"] not in conversation_history:
                conversation_history[transcription_output["uid"]] = []

            prompt = transcription_output['prompt'].strip()
            logging.info(f"[LLM INFO:] got prompt : {prompt}")
            
            # if prompt is same but EOS is True, we need that to send outputs to websockets
            if self.last_prompt == prompt:
                if self.last_output is not None and transcription_output["eos"]:
                    self.eos = transcription_output["eos"]
                    llm_queue.put({
                        "uid": transcription_output["uid"],
                        "llm_output": self.last_output,
                        "eos": self.eos,
                        "latency": self.infer_time
                    })
                    audio_queue.put({"llm_output": self.last_output, "eos": self.eos})
                    conversation_history[transcription_output["uid"]].append(
                        (transcription_output['prompt'].strip(), self.last_output[0].strip())
                    )
                    continue
            if prompt == "": continue

            input_text=self.format_prompt_chatml(prompt, conversation_history[transcription_output["uid"]], system_prompt=self.system_prompt)

            self.eos = transcription_output["eos"]
            encoded = encode_tokens(self.tokenizer, input_text, bos=False, device=device)
            prompt_length = encoded.size(0)
            print(prompt_length, encoded)
            t0 = time.perf_counter()

            y = self.generate(
                    encoded,
                    max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
            self.infer_time = time.perf_counter() - t0
            device_sync(device=device) # MKG

            output = self.tokenizer.decode(y.tolist()[prompt_length:], skip_special_tokens=True).strip()
            logging.info(f"[LLM INFO:] output: {output}")
            if output.startswith("<|im_start|>user"):
                output = output.split("<|im_start|>user")[1].strip()
            if output.startswith("<|im_start|>assistant"):
                output = output.split("<|im_start|>assistant")[1].strip()
            logging.info(f"[LLM INFO:] after splitting user, assistant output: {output}")
            output = [output.split("<|im_start|>")[0].strip()]

            logging.info(f"[LLM INFO:] Final output after splitting: {output[0]}")

            output = [get_llm_output_json(output[0])]

            if output is not None:
                self.last_output = output
                self.last_prompt = prompt
                llm_queue.put({
                    "uid": transcription_output["uid"],
                    "llm_output": output,
                    "eos": self.eos,
                    "latency": self.infer_time
                })
                
                logging.info(f"[LLM INFO:] Output: {output[0]}\nLLM inference done in {self.infer_time} ms\n\n")
                resp = None
                try:
                    resp = json.loads(output[0])['r']
                except Exception as e:
                    logging.warning(f"[LLM WARNING:] {e}")
                
                if resp is None:
                    resp = output[0].split('"r": ')[-1].split(', "m"')[0]
                audio_queue.put({"llm_output": [resp], "eos": self.eos})
                
            if self.eos:
                conversation_history[transcription_output["uid"]].append(
                    (transcription_output['prompt'].strip(), output[0].strip())
                )
                self.last_prompt = None
                self.last_output = None
            # tokens_generated = y.size(0) - prompt_length
            # print("tokens generated: ", tokens_generated)
            # print(output)
            # tokens_generated = y.size(0) - prompt_length
            # tokens_sec = tokens_generated / t
            # print(f"Time for inference : {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
            # print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__=="__main__":
    phi_engine = PhiEngine()
    phi_engine.run()
    
