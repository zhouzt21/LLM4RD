import ollama

import numpy as np 
import time
import logging 
import os
import hydra
import shutil
import re
import ast


def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()

def get_function_signature(code_string):
    # Parse the code string into an AST
    module = ast.parse(code_string)

    # Find the function definitions
    function_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]

    # If there are no function definitions, return None
    if not function_defs:
        return None

    # For simplicity, we'll just return the signature of the first function definition
    function_def = function_defs[0]

    input_lst = []
    # Construct the function signature (within object class)
    signature = function_def.name + '(self.' + ', self.'.join(arg.arg for arg in function_def.args.args) + ')'
    for arg in function_def.args.args:
        input_lst.append(arg.arg)
    return signature, input_lst


LLM4RD_ROOT_DIR = os.getcwd()
SMPLENV_ROOT_DIR = f"{LLM4RD_ROOT_DIR}/../smpl_envs"

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    # task = cfg.env.task
    task_description = cfg.env.description
    suffix = cfg.suffix   #??
    model = cfg.model

    # Loading all env, cfg and obs files
    env_name = cfg.env.env_name.lower()
    task_file = f'{LLM4RD_ROOT_DIR}/envs/{env_name}.py'
    task_obs_file = f'{LLM4RD_ROOT_DIR}/envs/{env_name}_obs.py'
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_code_string  = file_to_string(task_file)
    task_obs_code_string  = file_to_string(task_obs_file)
    output_file = f"{SMPLENV_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"


    # Loading all text prompts
    prompt_dir = f'{LLM4RD_ROOT_DIR}/utils/prompts'
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    # code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
    # policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    # execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

    in_context_learning_tip = file_to_string(f'{prompt_dir}/in_context_learning_tip.txt')
    input_rules = file_to_string(f'{prompt_dir}/input_rules.txt')

    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

    for iter in range(cfg.iteration):  # each iter provide only one code, and first produce K samples to do in context learning for the new sports
        logging.info(f"total iter: {cfg.iteration}, total sample: {cfg.sample}, current iter: {iter}")
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = cfg.sample if "gpt-3.5" in model else 4
        # logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

        while True:
            if total_samples >= cfg.sample:
                break
            for attempt in range(1000):
                try:
                    response_cur = ollama.chat(
                        model='llama3-en', 
                        messages=messages
                    )
                    content = response_cur['message']['content']
                    logging.info(f'{total_samples}, and the content is:{content}')
                    total_samples += 1 #chunk_size
                    break
                except Exception as e:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        # print("Current Chunk Size", chunk_size)
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)
            if response_cur is None:
                logging.info("Code terminated due to too many failed attempts!")
                exit()
            responses.append(response_cur) 

        code_runs = [] 
        output_context_string = ""
        for response_id in range(len(responses)):
            response_cur = responses[response_id]["message"]["content"]
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # Regex patterns to extract python code enclosed in response
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string
            # print(response_id, "\ncode_string: \n",code_string)

            # Remove unnecessary imports
            lines = code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])
                    
            # Add the Reward Signature to the environment code
            try:
                gpt_reward_signature, input_lst = get_function_signature(code_string)
            except Exception as e:
                logging.info(f"Iteration {iter}: Code Run {response_id} cannot parse function signature!")
                # continue

            code_runs.append(code_string)
            reward_signature = [
                f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",   #？？
                f"self.extras['gpt_reward'] = self.rew_buf.mean()",   #？？
                f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",  #？？
            ]
            indent = " " * 8
            reward_signature = "\n".join([indent + line for line in reward_signature])
            if "def compute_reward(self)" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_reward(self):", "def compute_reward(self):\n" + reward_signature)
            elif "def compute_reward(self, actions)" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature)
            else:
                raise NotImplementedError

            # Save the new environment code when the output contains valid code string!
            with open(output_file, 'w') as file:
                file.writelines(task_code_string_iter + '\n')
                file.writelines("from typing import Tuple, Dict" + '\n')
                file.writelines("import math" + '\n')
                file.writelines("import torch" + '\n')
                file.writelines("from torch import Tensor" + '\n')
                if "@torch.jit.script" not in code_string:
                    code_string = "@torch.jit.script\n" + code_string
                file.writelines(code_string + '\n')

            with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                file.writelines(code_string + '\n')

            # Copy the generated environment code to hydra output directory for bookkeeping
            shutil.copy(output_file, f"env_iter{iter}.py") #_response{response_id}

            output_context_string += code_string + "\n\n"
        
        # print("output_context_string:", output_context_string)
        with open(f'output{iter}.txt', 'a', encoding='utf-8') as file:
            file.write(output_context_string)        

        # generate new code (in context learning), iter once?
        in_context_learning_tip = in_context_learning_tip.format(input_rules=input_rules, output_context_string= output_context_string,
                                                                 task_obs_code_string=task_obs_code_string, task_description=task_description)
        # print("In-Context Learning Code:", in_context_learning_tip)
        messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": in_context_learning_tip}]

        for attempt in range(1000):
            try:
                response_cur_2 = ollama.chat(
                    model='llama3-en', 
                    messages=messages
                )
                content = response_cur_2['message']['content']
                logging.info(f'{content}')
                total_samples += chunk_size
                break
            except Exception as e:
                if attempt >= 10:
                    chunk_size = max(int(chunk_size / 2), 1)
                    # print("Current Chunk Size", chunk_size)
                logging.info(f"Attempt {attempt+1} failed with error: {e}")
                time.sleep(1)    
        if response_cur_2 is None:
            logging.info("Code terminated due to too many failed attempts!")
            exit()

        response_res = response_cur_2["message"]["content"]

        # Regex patterns to extract python code enclosed in response
        patterns = [
            r'```python(.*?)```',
            r'```(.*?)```',
            r'"""(.*?)"""',
            r'""(.*?)""',
            r'"(.*?)"',
        ]
        for pattern in patterns:
            code_string = re.search(pattern, response_res, re.DOTALL)
            if code_string is not None:
                code_string = code_string.group(1).strip()
                break
        code_string = response_res if not code_string else code_string

        # Remove unnecessary imports
        lines = code_string.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                code_string = "\n".join(lines[i:])
                
        # Add the Reward Signature to the environment code
        try:
            gpt_reward_signature, input_lst = get_function_signature(code_string)
        except Exception as e:
            logging.info(f"Iteration {iter}: Code Run final cannot parse function signature!")
            continue

        code_runs.append(code_string)
        reward_signature = [
            f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",   #？？
            f"self.extras['gpt_reward'] = self.rew_buf.mean()",   #？？
            f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",  #？？
        ]
        indent = " " * 8
        reward_signature = "\n".join([indent + line for line in reward_signature])
        if "def compute_reward(self)" in task_code_string:
            task_code_string_iter = task_code_string.replace("def compute_reward(self):", "def compute_reward(self):\n" + reward_signature)
        elif "def compute_reward(self, actions)" in task_code_string:
            task_code_string_iter = task_code_string.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature)
        else:
            raise NotImplementedError

        with open(f"env_iter{iter}_rewardonly_final.py", 'w') as file:
            file.writelines(code_string + '\n')


if __name__ == "__main__":
    main()
