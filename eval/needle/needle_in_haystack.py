"""
This script is adapted from 
https://github.com/gkamradt/LLMTest_NeedleInAHaystack

# GPT-4
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider OpenAI\
    --model_name gpt-4-1106-preview
    --api_key $OPENAI_API_KEY
) 2>&1  | tee logs/eval_gpt_4_128k.log

# LLaMA 2 32K. Remember to download the model first
(
CUDA_VISIBLE_DEVICES=2 python -u needle_in_haystack.py --s_len 0 --e_len 64000 --model_provider LLaMA --model_path /vepfs/wcf/G/zecheng/hf_models/llama-2-7b-80k -n 6 -t 2 --insert_short_key_id 1 --model_name_suffix test --shortcut_position 0 -tp
) 2>&1  | tee logs/eval_llama2_32k_instruct.log

# LongChat. Remember to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path /ML-A800/models/longchat-7b-v1.5-32k
) 2>&1  | tee logs/eval_longchat.log

# Our llama-2-7b-80k, requires 4*80G A100
# require you to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path ../../../llama-2-7b-80k
) 2>&1  | tee logs/eval_llama-2-7b-80k.log
"""

import tiktoken
import os 
import glob
import json
import tensor_parallel as tp
from transformers import AutoModelForCausalLM, AutoTokenizer
from anthropic import Anthropic
#from dotenv import load_dotenv
import numpy as np
import argparse
from rouge_score import rouge_scorer
from openai import OpenAI
from datetime import datetime, timezone
import time
import torch
from modelzipper.tutils import *

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

all_needles = auto_read_data("needle.jsonl")
shortcut_keys = auto_read_data("shortcut_key.jsonl")

def reset_rope(model, model_max_train_len, scaling_factor):
    for l in model.model.layers:
        l.self_attn.rotary_emb.scaling_factor = scaling_factor
        l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=model_max_train_len, device="cpu", dtype=torch.float32)
    return

"""
有哪些原因会导致捷径学习？
1. 变化的PPL -> 导致head过分关注某些内容
2. 训练数据泄露 -> 使用模型PPL大的sentence作为passkey还能检索出来吗
"""

class LLMNeedleHaystackTester:

    def __init__(self, haystack_dir="PaulGrahamEssays", results_version = 1, context_lengths_min = 1000, context_lengths_max = 128000, 
                 context_lengths_num_intervals = 40, context_lengths = None, document_depth_percent_min = 0, document_depth_percent_max = 100, 
                 insert_short_key_id = 0, document_depth_percent_intervals = 10, document_depth_percents = None, anthropic_api_key = None, 
                 document_depth_percent_interval_type = "linear", model_provider = "OpenAI", openai_api_key=None, model_name='', short_cut_strategy="random", 
                 model_name_suffix=None, num_concurrent_requests = 1, save_results = True, save_contexts = True, final_context_length_buffer = 200, 
                 seconds_to_sleep_between_completions = None, print_ongoing_status = True, ca_needle = 1, template_idx = 0, shortcut_position=0, tensor_parallel=True):
        """Functions

        Args:
            haystack_dir (str, optional): 实验地址路径. Defaults to "PaulGrahamEssays".
            retrieval_question (str, optional): [description]. Defaults to "What is the best thing to do in San Francisco?".
            results_version (int, optional): [description]. Defaults to 1.
            context_lengths_min (int, optional): [description]. Defaults to 1000.
            context_lengths_max (int, optional): [description]. Defaults to 128000.
            context_lengths_num_intervals (int, optional): [description]. Defaults to 40.
            context_lengths ([type], optional): [description]. Defaults to None.
            document_depth_percent_min (int, optional): [description]. Defaults to 0.
            document_depth_percent_max (int, optional): [description]. Defaults to 100.
            insert_short_key_id (int, optional): 插入捷径key的index(0 表示插入一个空的字符串进去). Defaults to 0.
            document_depth_percent_intervals (int, optional): [description]. Defaults to 10.
            document_depth_percents ([type], optional): [description]. Defaults to None.
            document_depth_percent_interval_type (str, optional): [description]. Defaults to "linear".
            model_provider (str, optional): [description]. Defaults to "OpenAI".
            openai_api_key ([type], optional): [description]. Defaults to None.
            anthropic_api_key ([type], optional): [description]. Defaults to None.
            model_name (str, optional): [description]. Defaults to ''.
            model_name_suffix ([type], optional): [description]. Defaults to None.
            num_concurrent_requests (int, optional): [description]. Defaults to 1.
            save_results (bool, optional): [description]. Defaults to True.
            save_contexts (bool, optional): [description]. Defaults to True.
            final_context_length_buffer (int, optional): [description]. Defaults to 200.
            seconds_to_sleep_between_completions ([type], optional): [description]. Defaults to None.
            print_ongoing_status (bool, optional): [description]. Defaults to True.
            ca_needle (int, optional): [description]. Defaults to 1.
            template_idx (int, optional): [description]. Defaults to 0.
            shortcut_position (int, optional): 插入shortcut的位置，是一个相对位置，0表示插在needle前面，1表示后面. Defaults to 0.

        Raises:
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
        """
    
        self.needle = all_needles[ca_needle-1]['value']
        self.shortcut_key = shortcut_keys[insert_short_key_id]['value']
        log_c(all_needles[ca_needle-1]['tag'])
        log_c(shortcut_keys[insert_short_key_id]['tag'])
        
        self.haystack_dir = haystack_dir
        self.retrieval_question = all_needles[ca_needle-1]['retrieval_question']
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.template_idx = template_idx
        self.shortcut_position = shortcut_position
        self.short_cut_strategy = short_cut_strategy
        self.testing_results = []

        if(self.model_provider not in ["OpenAI", "Anthropic"]):
            self.enc = AutoTokenizer.from_pretrained(model_name)
            log_c("loading from %s" % model_name)
            self.model_to_test = AutoModelForCausalLM.from_pretrained(model_name, use_flash_attention_2="flash_attention_2", torch_dtype=torch.bfloat16).eval()
            scaling_factor = 10 # hardcode
            reset_rope(self.model_to_test, model_max_train_len=81920, scaling_factor=scaling_factor)
            self.model_to_test = tp.tensor_parallel(self.model_to_test, sharded=True) if tensor_parallel else self.model_to_test.cuda()
        else: 
            self.model_to_test = OpenAI(api_key=openai_api_key)
            if(self.model_provider == "OpenAI"):
                self.enc = tiktoken.encoding_for_model(self.model_name)
            elif(self.model_provider == "Anthropic"):
                self.enc = Anthropic().get_tokenizer()
        
        self.needle_tok = self.enc(self.needle).input_ids[1:]
        self.shortcut_key_tok = self.enc(self.shortcut_key).input_ids[1:] if len(self.shortcut_key) > 0 else None

        if("/" in model_name):
            self.model_version = model_name.split("/")[-1]
        else: self.model_version = model_name
        if(model_name_suffix is not None): self.model_version += "_" + model_name_suffix

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        self.model_name = model_name
        self.model_to_test_description = model_name
        self.evaluation_model = None
        model_name = model_name.split('/')[-1]
        
    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0: return 0
        if x == 100: return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    
    def bound_evaluate_and_log(self, *args):
        self.evaluate_and_log(*args)

    def run_test(self, args): # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            if context_length < args.s_len or context_length > args.e_len: continue
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(context_length, depth_percent)

    def generate_prompt(self, context):
        # Generate the prompt for the Anthropic model
        # Replace the following line with the appropriate prompt structure
        if(self.model_provider not in ["OpenAI", "Anthropic"]):
            test_format_prefix = "<|im_start|> This is a very long story book: <book> "
            test_format_prefix_len = len(self.enc(test_format_prefix).input_ids) - 1  # skip the <s> token
            test_format1=f"{test_format_prefix}{context} </book>.\n Based on the content of the book, Question: {self.retrieval_question}\nAnswer: The best thing to do in San Francisco is"
            test_format2=f"{test_format_prefix}{context} </book>.\n Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
            return (test_format1, test_format_prefix_len) if self.template_idx == 1 else (test_format2, test_format_prefix_len)
        else: 
            return [{"role": "system","content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"},{"role": "user","content": context},{"role": "user","content": f"{self.retrieval_question} Don't give information outside the document or repeat your findings. The document definitely contains the answer, and I'm 100% sure. So try your best to find it."},{"role": "assistant","content":""}]

    def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                print("result exists, skipping")
                return
            else:
                print("result does not exist, testing")

        # Go generate the required length context and place your needle statement in
        context, insert_meta_data = self.generate_context(context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompt, test_format_prefix_len = self.generate_prompt(context)
        test_start_time = time.time()
        if(self.model_provider in ["OpenAI", "Anthropic"]):
            response = self.model_to_test.chat.completions.create(model=self.model_name, messages=prompt, max_tokens=300, temperature=0)
            response = response.choices[0].message.content
        else:
            prompt = self.enc(prompt, return_tensors="pt")
            input_ids = prompt['input_ids'].to(self.model_to_test.device)
            with torch.no_grad():
                outputs = self.model_to_test(input_ids)
                # need to find the key
                insert_st, insert_end = insert_meta_data["insert_point_bt"], insert_meta_data["insert_point_ed"]
                # add the prompt length
                st, end = insert_st + test_format_prefix_len, insert_end + test_format_prefix_len
                length = end - st + 1
                exp_st, exp_end = max(1, st - length), min(input_ids.size(-1), end + length) # expend st and end value to view wider positions
                shift_st, shift_end = st - exp_st, exp_end - end
                prefix_logits, needle_logits, suffix_logits = outputs.logits[0, exp_st-1:st-1, :], outputs.logits[0, st-1:end-1, :], outputs.logits[0, end-1:exp_end-1, :]
                prefix_labels, needle_labels, suffix_labels = input_ids[0,exp_st:st], input_ids[0,st:end], input_ids[0,end:exp_end]
                loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
                prefix_ppl, needle_ppl = torch.exp(loss_fct(prefix_logits, prefix_labels)), torch.exp(loss_fct(needle_logits, needle_labels))
                prefix_ppl, needle_ppl = prefix_ppl.item(), needle_ppl.item()
                if end != exp_end:                                     
                    suffix_ppl = torch.exp(loss_fct(suffix_logits, suffix_labels))
                    suffix_ppl = suffix_ppl.item()
                else:
                    suffix_ppl = 0.0  # if 0.0 then no suffix
                output_ids = self.model_to_test.generate(input_ids, max_new_tokens=50)
                response = self.enc.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        score = scorer.score(self.needle, response)['rouge1'].fmeasure*10
        
        results = {
            'model' : self.model_to_test_description,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'retrieval_question': self.retrieval_question,
            'needle' : self.needle,
            'shortcut_key' : self.shortcut_key if self.shortcut_key is not None else "",
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'),
            'ppls': [prefix_ppl, needle_ppl, suffix_ppl],
            "shift_st": shift_st, 
            "shift_end": shift_end,
        }
        results.update(insert_meta_data)
        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print (f"Response: {response}\n")

        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'

        if self.save_contexts:
            results['file_name'] = context_file_location

            # Save the context to file for retesting
            if not os.path.exists('contexts'):
                os.makedirs('contexts')

            if not os.path.exists(f'contexts/{self.model_version}'):
                os.makedirs(f'contexts/{self.model_version}')

            with open(f'contexts/{self.model_version}/{context_file_location}_context.txt', 'w') as f:
                f.write(context)
            
        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists('results'):
                os.makedirs('results')
            
            if not os.path.exists(f'results/{self.model_version}'):
                os.makedirs(f'results/{self.model_version}')

            # Save the result to file for retesting
            p = f'results/{self.model_version}/{context_file_location}_results.json'
            print("Writing at %s" % p)
            with open(p, 'w') as f:
                json.dump(results, f)

    def find_sublist(self, sub, bigger):
        bigger = bigger.cpu().tolist()[0]
        for i in range(len(bigger) - len(sub) + 1):
            if bigger[i:i+len(sub)] == sub:
                return i, i + len(sub)
        return None, None

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = 'results/' + self.model_version
        print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    # import ipdb; ipdb.set_trace()
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def generate_context(self, context_length, depth_percent): # Load up tiktoken so we navigate tokens more easily
        context = self.read_context_files() # Get your Paul Graham files loaded into a string
        context = self.encode_and_trim(context, context_length) # Truncate the Paul Graham essays to the context length you desire
        if self.shortcut_key_tok is not None:
            context, insert_meta_data = self.insert_needle_shortcut(context, depth_percent, context_length) 
        else: 
            context, insert_meta_data = self.insert_needle(context, depth_percent, context_length) 
        return context, insert_meta_data
    
    def encode_text_to_tokens(self, text):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.encode(text)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
    

    def insert_needle(self, context, depth_percent, context_length):  # just insert needle
        tokens_needle = self.needle_tok
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle + shortcut keys are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
        if(self.model_provider in ["LLaMA", "LongLLaMA"]): period_tokens = [29889, 869]
        elif(self.model_provider == "Mistral"): period_tokens = [842, 28723]
        elif(self.model_provider == "GLM"): period_tokens = [918, 30930]
        else: period_tokens = self.encode_text_to_tokens('.')

        if depth_percent == 100:
            tokens_new_context = tokens_context + tokens_needle
            insertion_point = len(tokens_new_context) - len(tokens_needle)
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print("insertion at %d" % insertion_point)
        
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context, {"shortcut_key_pos_bt": 0, "shortcut_key_pos_ed": 0, "insert_point_bt": insertion_point, "insert_point_ed": insertion_point+len(tokens_needle)}


    def insert_needle_shortcut(self, context, depth_percent, context_length):  # insert both shortcut and needle
        tokens_needle = self.needle_tok
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle + shortcut keys are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) + len(self.shortcut_key_tok) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle) - len(self.shortcut_key_tok)]

        # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
        if(self.model_provider in ["LLaMA", "LongLLaMA"]): period_tokens = [29889, 869]
        elif(self.model_provider == "Mistral"): period_tokens = [842, 28723]
        elif(self.model_provider == "GLM"): period_tokens = [918, 30930]
        else: period_tokens = self.encode_text_to_tokens('.')

        if depth_percent == 100:
            if self.short_cut_strategy == "random":
                shortcut_key_position = random.randint(self.final_context_length_buffer, len(tokens_context) - 1)
                tokens_new_context = tokens_context[:shortcut_key_position]
                # insert shortcut key in random position, before a whole sequence
                while tokens_new_context and tokens_new_context[-1] not in period_tokens:  
                    shortcut_key_position -= 1
                    tokens_new_context = tokens_context[:shortcut_key_position]
                tokens_new_context += self.shortcut_key_tok + tokens_context[:shortcut_key_position] + tokens_needle
            elif self.short_cut_strategy == "before":
                tokens_new_context = tokens_context + self.shortcut_key_tok + tokens_needle
            insertion_point = len(tokens_new_context) - len(tokens_needle)
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print("insertion at %d" % insertion_point)
        
            if self.short_cut_strategy == "random":
                tokens_new_context += tokens_needle + tokens_context[insertion_point:]
                if self.shortcut_position == 0: 
                    short_pos_st, short_pos_ed = self.final_context_length_buffer, insertion_point
                elif self.shortcut_position == 1: 
                    short_pos_st, short_pos_ed = insertion_point + len(tokens_needle), len(tokens_new_context) - 1
                
                if short_pos_st > short_pos_ed:  # can only insert shortcut key after the needle
                    self.shortcut_position = 1
                    short_pos_st, short_pos_ed = insertion_point + len(tokens_needle), len(tokens_new_context) - 1

                # Insert Shortcut squence
                shortcut_key_position = random.randint(short_pos_st, short_pos_ed)
                prefix, suffix = tokens_new_context[:shortcut_key_position], tokens_new_context[shortcut_key_position:]
                if self.shortcut_position == 0: # insert in the left, shift to left position 
                    while suffix and suffix[0] not in period_tokens:  # insert shortcut key before a whole sequence
                        shortcut_key_position -= 1 
                        prefix, suffix = tokens_new_context[:shortcut_key_position], tokens_new_context[shortcut_key_position:]
                else: # insert in the right, shift to right position  
                    while suffix and prefix[-1] not in period_tokens:  
                        shortcut_key_position += 1 
                        prefix, suffix = tokens_new_context[:shortcut_key_position], tokens_new_context[shortcut_key_position:]
                tokens_new_context = prefix + self.shortcut_key_tok + suffix
            elif self.short_cut_strategy == "before":
                tokens_new_context += self.shortcut_key_tok + tokens_needle + tokens_context[insertion_point:]
            elif self.short_cut_strategy == "after":
                tokens_new_context += self.shortcut_key_tok + tokens_needle + tokens_context[insertion_point:]
            else:
                raise NotImplementedError


        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context, {"shortcut_key_pos_bt": shortcut_key_position, "shortcut_key_pos_ed": shortcut_key_position + len(self.shortcut_key_tok), "insert_point_bt": insertion_point, "insert_point_ed": insertion_point+len(tokens_needle)}

    def get_context_length_in_tokens(self, context):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return len(self.enc.encode(context))
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            encoded = self.enc.encode(context)
            return len(self.enc.encode(context).ids)
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.encode(context)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(context).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.decode(tokens[:context_length])
        elif self.model_provider == "Anthropic":
            # Assuming you have a different decoder for Anthropic
            return self.enc.decode(tokens[:context_length])
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print (f"- Shortcut Key: {self.shortcut_key.strip()}")
        print ("\n\n")

    def start_test(self, args):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        #asyncio.run(self.run_test())
        self.run_test(args)


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--ca_needle', metavar='N', type=int, help='a number')
    parser.add_argument('-t', '--template_idx', metavar='N', type=int, help='a number')
    parser.add_argument('-s', '--s_len', metavar='N', type=int, help='a number')
    parser.add_argument('-e', '--e_len', metavar='N', type=int, help='a number')
    parser.add_argument('--shortcut_position', metavar='N', type=int, help='a number')
    parser.add_argument('--insert_short_key_id', metavar='N', type=int, help='a number')
    parser.add_argument('--model_path', type=str, default=None, help='path to model')
    parser.add_argument('--model_name', type=str, default=None, help='name of model')
    parser.add_argument('--model_name_suffix', type=str, default=None, help='name of model')
    parser.add_argument('--model_provider', type=str, default="LLaMA", help='which model to use') 
    parser.add_argument('--short_cut_strategy', type=str, default="random", help='how to insert the shortcut', choices=['random', 'before', 'after']) 
    parser.add_argument('--api_key', type=str, default="", help='OpenAI API Key')
    parser.add_argument('-tp', '--tensor_parallel', action='store_true', help='use tensor parallel')
    # parser = add_args(parser)
    args = parser.parse_args()

    if(args.model_path is not None):
        assert(args.model_name is None)
        model_name = args.model_path
    else: 
        assert(args.model_name is not None)
        model_name = args.model_name

    ht = LLMNeedleHaystackTester(
        model_name=model_name, 
        model_name_suffix=args.model_name_suffix,
        model_provider=args.model_provider,
        save_contexts=True,
        save_results=True,
        openai_api_key=args.api_key,
        ca_needle=args.ca_needle,
        template_idx=args.template_idx,
        insert_short_key_id=args.insert_short_key_id,
        shortcut_position=args.shortcut_position,
        tensor_parallel=args.tensor_parallel,
        short_cut_strategy=args.short_cut_strategy,
    )

    ht.start_test(args)
