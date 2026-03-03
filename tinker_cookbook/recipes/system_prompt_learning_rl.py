import logging
import time
from concurrent.futures import Future

import os
import re
import random
import json
import math

import numpy as np
import copy
from typing import List, Dict, Any
from collections import defaultdict

from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

import chz
import datasets
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

from json_utils import fix_json_backslashes, remove_json_comments, read_jsonl
from trueskill_utils import TrueSkillSystem, Rating, scores_to_ties

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/tinker-examples/system_prompt_learning_rl"
    model_name: str = "deepseek-ai/DeepSeek-V3.1"
    domain: str = "math"
    experiment_name: str = "system_prompt_learning_rl"
    dataset_size: int = 256
    dataset_pair: str = "aimo_beyondaime"  # "dapo_aime25", "aimo_beyondaime", "aime_and_amc_to_beyondaime", or "hmmt"

    train_mode: str = "evolution_rl"

    # RL
    n_epochs: int = 10
    batch_size: int = 10
    group_size: int = 5
    learning_rate: float = 4e-5
    max_length: int = 32768
    lora_rank: int = 32
    save_every: int = 20
    max_tokens: int = 15000
    sampling_temperature: float = 0.7
    top_p: float = 0.95
    max_update_operations: int = 2
    rl_loss_fn: str = "importance_sampling"

    # Evolution
    max_pool_size: int = 100
    choose_from_most_recent: int = 5
    num_parallel_programs: int = 3
    value_baseline_computation: str = "normal"

    test_group_size: int = 10
    resume_strategy: str = "best"

    mutation_sigma: float = 1.0
    crossover_sigma: float = 1.0
    evolution_sample_client: str = "reference"  # "reference" or "current"
    crossover_prob: float = 0.0
    program_selection_strategy: str = "uniform"  # "uniform", "lucb", "ucb", or "softmax"


def load_data(name: str) -> List[Dict[str, Any]]:
    if name == "AIME24":
        dataset = datasets.load_dataset("HuggingFaceH4/aime_2024", split="train")
        data = [{"problem": each["problem"], "groundtruth": each["answer"]} for each in dataset.to_list()]
        return data
    elif name == "AIME25":
        dataset = datasets.load_dataset("yentinglin/aime_2025", split="train")
        data = [{"problem": each["problem"], "groundtruth": each["answer"]} for each in dataset.to_list()]
        return data
    elif name == "DAPO-Math-17k":
        if os.path.exists("data/math/dataset/DAPO-Math-17k.json"):
            return json.load(open("data/math/dataset/DAPO-Math-17k.json"))
        else:
            dataset = datasets.load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
            data = dataset.to_list()
            transformed = {}
            for record in data:
                problem = record["prompt"][0]["content"].replace(
                    "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n",
                    "",
                ).replace('\n\nRemember to put your answer on its own line after "Answer:".', "")
                groundtruth = record["reward_model"]["ground_truth"]
                transformed[problem] = groundtruth
            random.seed(42)
            transformed = [{"problem": k, "groundtruth": v} for k, v in transformed.items()]
            random.shuffle(transformed)
            os.makedirs("data/math/dataset", exist_ok=True)
            json.dump(transformed, open("data/math/dataset/DAPO-Math-17k.json", "w"), indent=2)
            return transformed
    elif name == "aimo-validation-aime":
        dataset = datasets.load_dataset("AI-MO/aimo-validation-aime", split="train")
        data = [{"problem": each["problem"], "groundtruth": each["answer"]} for each in dataset.to_list()]
        return data
    elif name == "BeyondAIME":
        dataset = datasets.load_dataset("ByteDance-Seed/BeyondAIME", split="test")
        data = [{"problem": each["problem"], "groundtruth": each["answer"]} for each in dataset.to_list()]
        return data
    elif name == "aimo-validation-amc":
        dataset = datasets.load_dataset("AI-MO/aimo-validation-amc", split="train")
        data = [{"problem": each["problem"], "groundtruth": each["answer"]} for each in dataset.to_list()]
        return data
    elif name == "aime_and_amc":
        # Combine aimo-validation-aime and aimo-validation-amc
        aime_dataset = datasets.load_dataset("AI-MO/aimo-validation-aime", split="train")
        aime_data = [{"problem": each["problem"], "groundtruth": each["answer"]} for each in aime_dataset.to_list()]
        amc_dataset = datasets.load_dataset("AI-MO/aimo-validation-amc", split="train")
        amc_data = [{"problem": each["problem"], "groundtruth": each["answer"]} for each in amc_dataset.to_list()]
        all_data = aime_data + amc_data
        random.seed(42)
        random.shuffle(all_data)
        return all_data
    elif name == "hmmt_train":
        # Combine hmmt_feb_2023, hmmt_feb_2024, hmmt_feb_2025
        all_data = []
        for year in ["2023", "2024", "2025"]:
            dataset = datasets.load_dataset(f"MathArena/hmmt_feb_{year}", split="train")
            data = [{"problem": each["problem"], "groundtruth": each["answer"]} for each in dataset.to_list()]
            all_data.extend(data)
        random.seed(42)
        random.shuffle(all_data)
        return all_data
    elif name == "hmmt_nov_2025":
        dataset = datasets.load_dataset("MathArena/hmmt_nov_2025", split="train")
        data = [{"problem": each["problem"], "groundtruth": each["answer"]} for each in dataset.to_list()]
        return data
    else:
        raise NotImplementedError(f"Dataset {name} not supported")


def get_dataset_pair(pair_name: str) -> tuple[str, str]:
    """
    Returns (train_dataset_name, test_dataset_name) for a given pair name.
    """
    pairs = {
        "dapo_aime25": ("DAPO-Math-17k", "AIME25"),
        "aimo_beyondaime": ("aimo-validation-aime", "BeyondAIME"),
        "aime_and_amc_to_beyondaime": ("aime_and_amc", "BeyondAIME"),
        "hmmt": ("hmmt_train", "hmmt_nov_2025"),
    }
    if pair_name not in pairs:
        raise ValueError(f"Unknown dataset_pair: {pair_name}. Available: {list(pairs.keys())}")
    return pairs[pair_name]


def verify_func(model_output, ground_truth) -> float:
    verify_function = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0
    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + str(ground_truth) + "}"
    try:
        ret_score, _ = verify_function([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = 0
    return float(ret_score)


def pass_at_k(n: int, c: int, k: int):
    """
    Calculate pass@k metric using the unbiased estimator.
    
    Args:
        n: Total number of samples generated per problem
        c: Number of correct samples (samples that pass all tests)
        k: Number of samples to consider
        
    Returns:
        pass@k probability
        
    Formula: pass@k = 1 - C(n-c, k) / C(n, k)
    Computed as: 1 - prod((n-c-i)/(n-i) for i in range(k))
    """
    if n - c < k:
        return 1.0
    
    # Numerically stable computation
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    
    return 1.0 - result


def pass_at_k_for_range(n: int, c: int):
    results = []
    for k in range(1, n + 1):
        results.append(pass_at_k(n, c, k))
    return results


def get_operations_from_json(llm_response):
    # ADVICE: This JSON extraction is intentionally simple but brittle.
    # - If the model does not emit a ```json fenced block (or uses ```JSON), `split` may grab the wrong span.
    # - If multiple fenced blocks exist, this will always pick the last ```json block.
    # Consider using a case-insensitive regex to extract the fenced JSON, and/or fallback heuristics.
    response_json = llm_response.split("```json")[-1].split("```")[0]
    try:
        operations = json.loads(fix_json_backslashes(response_json))
    except json.decoder.JSONDecodeError:
        try:
            operations = json.loads(remove_json_comments(fix_json_backslashes(response_json)))
        except json.decoder.JSONDecodeError:
            print("Encountered json.decoder.JSONDecodeError ...")
            print("------------------ DEBUG ------------------")
            print("------------------ llm_response BEGINS ------------------")
            print(llm_response)
            print("------------------ llm_response ENDS ------------------\n")
            print("------------------ JSON RESPONSE BEGINS ------------------")
            print(response_json)
            print("------------------ JSON RESPONSE ENDS ------------------")
            print("------------------ DEBUG ------------------")
            operations = []
    return operations


PROBLEM_WITH_PRINCIPLE_TEMPLATE = """Please solve the following problem:
{problem}

When solving problems, you MUST first carefully read and understand the helpful instructions and principles below:
{principles}"""


SINGLE_ROLLOUT_SUMMARY_TEMPLATE = """An agent system was provided with some principles, and then it produced the following trajectory to solve the given problem. Please summarize the trajectory step by step:

1. For each step, describe **what action is being taken**, and which principle has been used in this step.
2. Given the grading of this rollout and the correct answer, identify and explain any steps that **represent detours, errors, or backtracking**, highlighting why they might have occurred and what impact they had on the trajectory's progress. 
3. Maintain **all the core outcomes of each step**, even if they were part of a flawed process.

<trajectory>
{trajectory}
</trajectory>

<evaluation>
{grade}
</evaluation>

<groundtruth>
{answer}
</groundtruth>

Only return the trajectory summary of each step, e.g.,
1. what happened in the first step and the core outcomes
2. what happened in the second step and the core outcomes
3. ..."""


SINGLE_QUERY_CRITIQUE_TEMPLATE = """An agent system is provided with a set of principles and has tried to solve the problem multiple times, with both successful and incorrect solutions. Review these problem-solving attempts and extract generalizable principles. Follow these steps:

1. Trajectory Analysis:
    - For successful steps: Identify key correct decisions and insights
    - For errors: Pinpoint where and why the reasoning went wrong
    - Note any important patterns or strategies used or missed
    - Review why some trajectories failed. Were any existing principles missed, or did the principles not provide enough guidance?

2. Update Existing Principles
    - Some trajectories may be correct and others may be wrong; you should ensure there are principles that can help achieve correct solutions
    - You can use two types of operations: ["modify", "add"]
        * "modify": You can modify a current principle to make it more helpful
        * "add": You can introduce a new principle that may be needed
    - You can make at most {max_operations} updates with clear, generalizable lessons for this case
    - Before updating each principle, you need to:
        * Specify when it would be most relevant
        * List key problem features that make this principle applicable
        * Identify similar problem patterns where this principle applies

3. Requirements for Each Principle That Is Modified or Added
    - Begin with a brief general background (a few words) in the principle
    - Focus on strategic thinking patterns, not specific calculations
    - Emphasize decision points that could be applied to similar problems

Please provide reasoning in detail under the guidance of the above three steps.
After the step-by-step reasoning, you will finish by returning a list of operations in the following JSON format:
```json
[
    {{
        "operation": "modify",
        "principle": "the modified principle",
        "modified_from": "G17"  # specify the ID of the principle being modified
    }},
    {{
        "operation": "add",
        "principle": "the added principle",
    }},
    ...
]
```
Your proposed updates do not necessarily need to use both types of operations; using only one type of operation is good too.

<problem>
{problem}
</problem>

<trajectories>
{trajectories}
</trajectories>

<groundtruth>
{answer}
</groundtruth>

<principle>
{principles}
</principle>"""


BATCH_PRINCIPLE_UPDATE_TEMPLATE = """An agent system is provided with a set of principles, and has tried to solve problems with those principles. Some suggestions have been made to improve the existing principles.
Your task is to collect and review these individual suggestions, and come up with a final Revision Plan for the existing principles.

Each final principle (produced after the final Revision Plan) must satisfy the following requirements:
1. It must be clear, generalizable lessons for this case, with no more than 32 words
2. Begin with general background with several words in the principle
3. Focus on strategic thinking patterns, not specific calculations
4. Emphasize decision points that could apply to similar problems
5. Avoid repeating similar sayings or ideas across multiple principles

<existing_principles>
{principles}
</existing_principles>

<suggested_updates>
{updates}
</suggested_updates>

Please provide reasoning in each suggestion, and think about how to update the existing principles.
You can use two types of operations: ["modify", "merge"]
* "modify": You can modify an existing principle to make it more helpful
* "merge": You can merge similar principles into a more general form to reduce duplication

After generating the step-by-step reasoning, you will finish by providing the final Revision Plan as a list of operations in the following JSON format:
```json
[
    {{
        "operation": "modify",
        "principle": "the modified principle",
        "modified_from": "C1"  # specify the str ID of the principle being modified
    }},
    {{
        "operation": "merge",
        "principle": "the merged principle",
        "merged_from": ["C1", "C3", "S4", ...]  # specify the str IDs of the principles being merged; at least two IDs are needed
    }},
    ...
]
```

Your final Revision Plan does not necessarily need to use both types of operations; using only one type of operation is good too."""


CROSSOVER_PROMPT_START = """You are given multiple sets of principles that were independently evolved to guide an agent system in solving problems.
Principle Set A is the primary set that we want to improve by learning from the strengths of the other principle sets.

"""


CROSSOVER_ANALYSIS_PROMPT = """Each principle set listed above strictly outperformed all others on at least one question. Your task is to analyze this evidence and improve Principle Set A.

Follow these steps:

1. Evidence-Based Analysis:
   - For each question, explain WHY the best-performing set succeeded. Consider:
     * Did it have specific principle(s) that were relevant and helpful? If not, did it succeed because other sets had counterproductive principles on this topic? How likely was the success due to chance, or due to targeted guidance by specific principles?
   - Cite principle IDs (e.g., A.G3, B.G7) and question IDs (e.g., Q.0, Q.5) where applicable.

2. What to Preserve and What to Change in Set A:
   - For each question where Set A performed best: Which Set A principles likely contributed to its success? These should be preserved.
   - For each question where other sets performed best: Did they have useful principles Set A lacked? Or did Set A have a principle that was relevant to this question type but led to worse reasoning? Consider whether the difference is meaningful or likely due to chance.
   - Note: Principles that are simply irrelevant to the current batch of questions should be left alone - they may be useful for other problem types.

3. Propose Evidence-Grounded Improvements:
   - Think about how to improve Principle Set A. The improvement plan should be supported by evidence from the questions above, but use your judgment when the evidence is indirect.
   - Any additions or modifications to Set A should be based on or inspired by successful principles from other sets, not invented from scratch.
   - Note: Do not output JSON during your reasoning. Only output JSON once, at the very end.

You can use three types of operations to improve Principle Set A:
* "add": Introduce a new principle to Principle Set A (inspired by principles that helped other sets)
* "modify": Modify an existing principle in Principle Set A
* "remove": Remove a principle from Principle Set A that appears to be counterproductive

Please provide detailed reasoning following the above three steps, with explicit citations to principle IDs and question IDs throughout.

After the step-by-step reasoning, return a list of operations in the following JSON format:
```json
[
    {
        "operation": "add",
        "evidence": "Set B performed best on Q.3 and Q.7 using B.G5, which addresses X that Set A lacks",
        "principle": "the new principle to add"
    },
    {
        "operation": "modify",
        "modified_from": "A.G17",
        "evidence": "A.G17 helped on Q.1, but Set B performed best on Q.4 which needed stronger guidance on Y",
        "principle": "the modified principle"
    },
    {
        "operation": "remove",
        "remove_id": "A.G5",
        "evidence": "Set B performed best on Q.8 and Q.12 where A.G5 likely misled the reasoning"
    },
    ...
]
```

You do not need to use all operation types. Only propose operations that are clearly supported by the evidence. If the evidence does not strongly support any changes, it is acceptable to return an empty list.
"""


def build_crossover_prompt(system_prompt_list, problems_each_prompt_did_best):
    """
    Build a crossover prompt to improve Principle Set A using insights from other sets.
    
    Args:
        system_prompt_list: List of principle dicts. The first (index 0) is Set A (reference to improve).
        problems_each_prompt_did_best: List of problem lists that each principle set performed best on.
    
    Returns:
        A formatted prompt string for the LLM.
    """
    prompt = CROSSOVER_PROMPT_START
    
    for prompt_idx, system_prompt in enumerate(system_prompt_list):
        prompt_label = chr(ord('A') + prompt_idx)
        prompt += f"Principle Set {prompt_label}:\n"
        prompt += "<principles>\n"
        prompt += "\n".join([f"[{prompt_label}.{i}]: {e}" for i, e in system_prompt.items()])
        prompt += "\n</principles>\n\n"
    
    prompt += "Below are the questions each principle set performed best on.\n"
    for prompt_idx, questions in enumerate(problems_each_prompt_did_best):
        prompt_label = chr(ord('A') + prompt_idx)
        prompt += f"\nQuestions that Principle Set {prompt_label} did best on:\n"
        prompt += "<questions>\n"
        prompt += "\n".join([f"Q.{q_idx}: {question.strip()}" for q_dict in questions for q_idx, question in q_dict.items()])
        prompt += "\n</questions>\n"
    
    prompt += "\n" + CROSSOVER_ANALYSIS_PROMPT
    return prompt


def process_operations(operations, principles_to_modify):
    max_id = max((int(k[1:]) for k in principles_to_modify.keys()), default=-1)
    print(f"Processing {len(operations)} operations from JSON ...")
    for op in operations:
        try:
            if op["operation"] == "add":
                max_id += 1
                principles_to_modify[f"G{max_id}"] = op["principle"]
            elif op["operation"] == "modify":
                # modified_from is like "A.G17", extract "G17"
                principle_id = op["modified_from"].split(".")[-1]
                if principle_id in principles_to_modify:
                    principles_to_modify[principle_id] = op["principle"]
            elif op["operation"] == "remove":
                # remove_id is like "A.G5", extract "G5"
                principle_id = op["remove_id"].split(".")[-1]
                if principle_id in principles_to_modify:
                    del principles_to_modify[principle_id]
        except Exception as e:
            print(f"Error applying crossover operation: {op} | {e}")
    
    result_principles = {f"G{i}": p for i, p in enumerate(principles_to_modify.values())}
    return result_principles


def get_sample_future(conv, renderer, sampling_client, sampling_params):
    model_input = renderer.build_generation_prompt(conv)
    sample_future = sampling_client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    )
    return sample_future


def get_text_from_sampled_future(sample_future, renderer):
    sample_result = sample_future.result()
    sampled_tokens = sample_result.sequences[0].tokens
    parsed_message, _ = renderer.parse_response(sampled_tokens)
    msg_text = parsed_message["content"]
    return msg_text


def llm_chat(conv, renderer, sampling_client, sampling_params):
    sample_future = get_sample_future(conv, renderer, sampling_client, sampling_params)
    return get_text_from_sampled_future(sample_future, renderer)


def load_rollouts(rollout_filename: str) -> list[dict]:
    results = []
    if os.path.exists(rollout_filename):
        with open(rollout_filename, encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
    else:
        print("Did not find existing rollouts in:", rollout_filename)
    return results


def save_rollouts(results: list[dict], rollout_filename: str):
    with open(rollout_filename, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


class PrincipleUpdater:
    def __init__(self, renderer, sampling_client, sampling_params, max_operations=1):
        self.renderer = renderer
        self.sampling_client = sampling_client
        self.sampling_params = sampling_params
        self.max_operations = max_operations

    def run(self, rollouts, principles, save_dir):
        # 1. Summarize trajectory for each rollout
        problem_to_summarized_rollouts = self._single_rollout_summary(
            rollouts=rollouts,
            save_dir=save_dir,
        )

        # 2. Generate critique for each query
        critiques = self._single_query_critique(
            problem_to_summarized_rollouts=problem_to_summarized_rollouts,
            principles=principles,
            save_dir=save_dir,
            max_operations=self.max_operations,
        )

        # 3. batch update principles
        new_principles = self._batch_update(
            principles=principles,
            critiques=critiques,
            save_dir=save_dir,
        )

        # 4. assign new principle IDs
        new_principles = {
            f"G{i}": exp for i, exp in enumerate(new_principles.values())
        }
        return new_principles

    def _single_rollout_summary(self, rollouts, save_dir):
        # check file existence
        filename = os.path.join(save_dir, "single_rollout_summary.json")
        if os.path.exists(filename):
            print("Overwriting:", filename)

        # group by problems
        problems_to_rollouts = defaultdict(list)
        for each in rollouts:
            if "trajectories" in each and len(each["trajectories"]) > 0:
                problems_to_rollouts[each["problem"]].append(each)

        results = defaultdict(list)
        all_rollouts_to_process = []
        for rollouts in problems_to_rollouts.values():
            scores = [each["reward"] for each in rollouts]
            avg_score = sum(scores) / max(len(scores), 1)
            # ADVICE: This filter only keeps problems with mixed outcomes (some correct, some wrong).
            # That can easily starve the updater when the model is very weak (all 0) or very strong (all 1).
            # If you want learning signals from failures, consider also processing all-wrong cases.
            if 0 < avg_score < 1:
                all_rollouts_to_process.extend(rollouts)

        sample_futures_list = []
        for cur in all_rollouts_to_process:
            prompt = SINGLE_ROLLOUT_SUMMARY_TEMPLATE.format(
                trajectory=cur["trajectories"][0]["trajectory"],
                grade="This trajectory delivers **" + ("correct" if cur["reward"] else "wrong") + "** answer",
                answer=cur["groundtruth"]
            )
            conversation = [{"role": "user", "content": prompt}]
            sample_futures_list.append(
                get_sample_future(conversation, self.renderer, self.sampling_client, self.sampling_params)
            )

        for cur, sample_future in zip(all_rollouts_to_process, sample_futures_list):
            traj_summary = get_text_from_sampled_future(sample_future, self.renderer)
            result = {"trajectory_summary": traj_summary, **cur}
            problem = cur["problem"]
            results[problem].append(result)

        # write to file
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _single_query_critique(self, problem_to_summarized_rollouts, principles, save_dir, max_operations=1):
        # check file existence
        filename = os.path.join(save_dir, "single_query_critique.json")
        if os.path.exists(filename):
            print("Overwriting:", filename)

        all_rollouts = []
        for rollouts in problem_to_summarized_rollouts.values():
            scores = [each["reward"] for each in rollouts]
            avg_score = sum(scores) / len(scores)
            # ADVICE: Same starvation risk as above: only mixed-success problems are critiqued.
            # Consider handling avg_score==0 (all wrong) to propose new principles, and avg_score==1 (all right)
            # to optionally merge/deduplicate principles.
            if 0 < avg_score < 1:
                all_rollouts.append(rollouts)

        sample_futures_list = []
        rollouts_per_problem_list = []
        for rollouts_per_problem in all_rollouts:
            problem = rollouts_per_problem[0]["problem"]
            answer = rollouts_per_problem[0]["groundtruth"]
            strings_to_join = []
            for i, each in enumerate(rollouts_per_problem):
                string_id = i + 1
                answer_correctness = 'correct' if each["reward"] else 'wrong'
                traj_summary_i = each['trajectory_summary']
                string_i = f"Trajectory {string_id} (Answer {answer_correctness}):\n{traj_summary_i}"
                strings_to_join.append(string_i)
            formatted_trajectories = "\n\n".join(strings_to_join)
            formatted_principles = "\n".join([ f"[{i}]. {e}" for i, e in principles.items() ]) if principles else "None"
            prompt = SINGLE_QUERY_CRITIQUE_TEMPLATE.format(
                max_operations=max_operations,
                problem=problem,
                trajectories=formatted_trajectories,
                answer=answer,
                principles=formatted_principles,
            )
            conversation = [{"role": "user", "content": prompt}]
            sample_futures_list.append(
                get_sample_future(conversation, self.renderer, self.sampling_client, self.sampling_params)
            )
            # IMPORTANT: keep a 1:1 mapping between each Future and its corresponding rollouts.
            # Otherwise, referencing `rollouts_per_problem` later would accidentally reuse the last loop value.
            rollouts_per_problem_list.append(rollouts_per_problem)

        results = []
        for rollouts_per_problem, sample_future in zip(rollouts_per_problem_list, sample_futures_list):
            raw_response = get_text_from_sampled_future(sample_future, self.renderer)
            # parse json
            operations = get_operations_from_json(raw_response)
            result = {"rollouts": rollouts_per_problem, "critique": raw_response, "operations": operations[:max_operations]}
            results.append(result)

        # write results
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _batch_update(self, principles, critiques, save_dir, max_retries=3):
        print("Batch update")
        filename = os.path.join(save_dir, "batch_update.json")
        if os.path.exists(filename):
            print("Overwriting:", filename)

        # collect operations
        all_operations = []
        for each in critiques:
            all_operations.extend(each["operations"])
        print("- Num of operations to process:", len(all_operations))

        # split principles
        candidate_principles = copy.deepcopy(principles)
        to_modify = []
        max_ID = 0
        # ADVICE: ID-collision risk.
        # `max_ID` starts at 0 and new principles are keyed as C0, C1, ...
        # If `candidate_principles` already contains C* keys (from prior runs), this may overwrite existing ones.
        # Safer: initialize max_ID from existing keys, e.g. max(int(k[1:]) for k in candidate_principles if k.startswith('C')) + 1.
        for operation in all_operations:
            if operation["operation"] == "modify":
                if operation["modified_from"] in candidate_principles:
                    to_modify.append(operation)
            elif operation["operation"] == "add":
                candidate_principles[f"C{max_ID}"] = operation["principle"]
                max_ID += 1

        print("- Num of added principles:", max_ID)
        print("- Num of principles to be modified:", len(to_modify))
        print("- Num of candidate principles:", len(candidate_principles))

        # use LLM to get the revision plan
        revision_prompt = BATCH_PRINCIPLE_UPDATE_TEMPLATE.format(
            principles=candidate_principles, 
            updates=to_modify
        )
        response = llm_chat(
            [{"role": "user", "content": revision_prompt}],
            self.renderer,
            self.sampling_client,
            self.sampling_params,
        )
        revision_plan = get_operations_from_json(response)

        # modify candidate principles
        new_principles = copy.deepcopy(candidate_principles)
        for operation in revision_plan:
            try:
                if operation["operation"] == "modify":
                    new_principles[operation["modified_from"]] = operation["principle"]
                elif operation["operation"] == "merge":
                    for principle_ID in operation["merged_from"]:
                        if principle_ID not in new_principles:
                            raise Exception(f"ID {principle_ID} not found for merging")
                    for principle_ID in operation["merged_from"]:
                        if principle_ID in new_principles:
                            del new_principles[principle_ID]
                    new_principles[f"C{max_ID}"] = operation["principle"]
                    max_ID += 1
            except Exception as e:
                print("Error: failed to complete principle update:", operation, "|", e)
        print("- Num of revised candidate principles:", len(new_principles))

        # write to file
        with open(filename, "w") as f:
            json.dump(
                {
                    "operations": all_operations,
                    "response": response,
                    "revision_plan": revision_plan,
                    "new_principles": new_principles,
                },
                f,
                indent=2,
            )
        return new_principles


def chunk_groups(principles_pool, principle_group_size):
    """
    Split `principles_pool` into non-overlapping groups (chunks),
    each with size at most `principle_group_size`, preserving order.

    Example:
        chunk_groups([1,2,3,4,5], 2) -> [[1,2],[3,4],[5]]
    """
    if principle_group_size <= 0:
        raise ValueError("principle_group_size must be a positive integer")

    pool = list(principles_pool)  # in case an iterator is passed
    return [pool[i:i + principle_group_size] for i in range(0, len(pool), principle_group_size)]


FORMAT_SYMBOL = "\\boxed{}"


def calculate_mutation_rating(rating: Rating, mutation_sigma: float = 2.0):
    return rating.mu, math.sqrt(rating.sigma ** 2 + mutation_sigma ** 2)


def calculate_crossover_rating(rating_list: List[Rating], crossover_sigma: float = 0.5):
    precision_list = [rating.precision for rating in rating_list]
    precision_sum = sum(precision_list)
    weighted_mu = sum([rating.precision_mean for rating in rating_list]) / precision_sum
    weighted_sigma = math.sqrt(1 / precision_sum + crossover_sigma ** 2)
    return weighted_mu, weighted_sigma


class Program:
    def __init__(self, principles, program_id, self_modified_from=-1, parents_list=None, mu=25.0, sigma=25.0/3, timestep=None):
        self.principles = principles
        self.program_id = program_id
        self.timestep = timestep
        # mutation
        self.self_modified_from = self_modified_from
        self.self_modify_id_list = []
        # crossover
        self.parents_list = parents_list
        self.children_list = []
        # metrics
        self.past_score_history = []
        self.rating = Rating(mu=mu, sigma=sigma)
    
    def add_self_modify(self, self_modify_id):
        self.self_modify_id_list.append(self_modify_id)
    
    def add_child(self, child_id):
        self.children_list.append(child_id)
    
    def add_eval_score(self, score):
        self.past_score_history.append(score)
    
    def update_rating(self, new_rating):
        self.rating = Rating(mu=new_rating.mu, sigma=new_rating.sigma)

    def get_rating(self, strategy="mean"):
        if strategy == "mean":
            return self.rating.mu
        elif strategy == "explore":
            return self.rating.conservative_rating(factor=-2.0)
        elif strategy == "exploit":
            return self.rating.conservative_rating(factor=2.0)
        else:
            raise NotImplementedError(f"strategy {strategy} is not implemented")

    def state_dict(self):
        return {
            "principles": self.principles,
            "program_id": self.program_id,
            "timestep": self.timestep,
            "self_modified_from": self.self_modified_from,
            "self_modify_id_list": self.self_modify_id_list,
            "parents_list": self.parents_list,
            "children_list": self.children_list,
            "past_score_history": self.past_score_history,
            "rating": self.rating.state_dict()
        }
    
    def load_state_dict(self, state):
        self.principles = state["principles"]
        self.program_id = state["program_id"]
        self.past_score_history = state["past_score_history"]
        self.timestep = state.get("timestep", self.timestep)
        self.self_modified_from = state.get("self_modified_from", self.self_modified_from)
        self.self_modify_id_list = state.get("self_modify_id_list", self.self_modify_id_list)
        self.parents_list = state.get("parents_list", self.parents_list)
        self.children_list = state.get("children_list", self.children_list)
        if "rating" in state:
            self.rating.load_state_dict(state["rating"])


class EvolutionPool:
    def __init__(self, max_pool_size):
        self.max_pool_size = max_pool_size
        self.max_program_id = -1
        self.evolution_pool = []
    
    def get_new_program_id(self):
        self.max_program_id += 1
        return self.max_program_id
    
    def add_program_to_pool(self, program):
        self.evolution_pool.append(program)
        if len(self.evolution_pool) > self.max_pool_size:
            self.evolution_pool.pop(0)
    
    def sample_program_uniform(self, pool_size):
        indices = list(range(len(self.evolution_pool)))
        candidate_indices = indices[-pool_size:] if pool_size < len(indices) else indices
        sampled_idx = random.choice(candidate_indices)
        print("         - Choosing from candidate indices in evolution pool:", candidate_indices, "index:", sampled_idx, "program_id:", self.evolution_pool[sampled_idx].program_id)
        return self.evolution_pool[sampled_idx]
    
    def sample_k_programs_uniform(self, pool_size, k, include_last=True):
        indices = list(range(len(self.evolution_pool)))
        candidates = indices[-pool_size:] if pool_size < len(indices) else indices
        
        if include_last and len(candidates) > k:
            last = [candidates[-1]]
            rest = candidates[:-1]
            if k - 1 <= len(rest):
                sampled = random.sample(rest, k - 1)
            elif rest:
                sampled = random.choices(rest, k=k - 1)
            else:
                sampled = []  # pool_size=1, just return the last element
            return [self.evolution_pool[i] for i in last + sampled]
        
        # Original behavior
        if k <= len(candidates):
            return [self.evolution_pool[i] for i in random.sample(candidates, k)]
        return [self.sample_program_uniform(pool_size) for _ in range(k)]

    def sample_k_programs_lucb(self, pool_size, k, include_last=True):
        """
        Select k programs using LUCB (Lower/Upper Confidence Bound) strategy:
        - 1 champion: the program with the best "exploit" score (highest lower bound)
        - k-1 challengers: programs with the best "explore" scores (highest upper bound)
        - If include_last=True, ensure the most recent program is among challengers

        Args:
            pool_size: Number of recent programs to consider as candidates
            k: Number of programs to select
            include_last: If True, always include the most recent program

        Returns:
            List of Program objects (up to k programs)
        """
        if k <= 0 or len(self.evolution_pool) == 0:
            return []

        indices = list(range(len(self.evolution_pool)))
        candidates = indices[-pool_size:] if pool_size < len(indices) else indices

        if len(candidates) == 0:
            return []

        # Find the champion: best "exploit" score (highest lower bound)
        exploit_scores = [
            (idx, self.evolution_pool[idx].get_rating(strategy="exploit"))
            for idx in candidates
        ]
        exploit_scores.sort(key=lambda x: x[1], reverse=True)
        champion_idx = exploit_scores[0][0]

        # Find challengers: top k-1 by "explore" score (highest upper bound), excluding champion
        remaining_candidates = [idx for idx in candidates if idx != champion_idx]
        explore_scores = [
            (idx, self.evolution_pool[idx].get_rating(strategy="explore"))
            for idx in remaining_candidates
        ]
        explore_scores.sort(key=lambda x: x[1], reverse=True)

        # Get top k-1 challenger indices
        num_challengers = min(k - 1, len(explore_scores))
        challenger_indices = [idx for idx, score in explore_scores[:num_challengers]]

        # If include_last, ensure the last program is among challengers
        if include_last and len(candidates) > 0:
            last_idx = candidates[-1]
            if last_idx != champion_idx and last_idx not in challenger_indices:
                if len(challenger_indices) >= k - 1 and len(challenger_indices) > 0:
                    # Replace the lowest-ranked challenger with the last one
                    challenger_indices[-1] = last_idx
                elif len(challenger_indices) < k - 1:
                    challenger_indices.append(last_idx)

        result_indices = [champion_idx] + challenger_indices
        # Pad with last program duplicates if not enough unique programs
        last_idx = candidates[-1]
        while len(result_indices) < k:
            result_indices.append(last_idx)

        return [self.evolution_pool[i] for i in result_indices]

    def sample_k_programs_ucb(self, pool_size, k, include_last=True):
        """
        Select k programs using UCB (Upper Confidence Bound) strategy:
        - Rank all candidates by explore score (UCB = mu + 2*sigma)
        - Take top-k programs

        Args:
            pool_size: Number of recent programs to consider as candidates
            k: Number of programs to select
            include_last: If True, always include the most recent program

        Returns:
            List of Program objects (up to k programs)
        """
        if k <= 0 or len(self.evolution_pool) == 0:
            return []

        indices = list(range(len(self.evolution_pool)))
        candidates = indices[-pool_size:] if pool_size < len(indices) else indices

        if len(candidates) == 0:
            return []

        # Rank all candidates by explore score (UCB = mu + 2*sigma)
        explore_scores = [
            (idx, self.evolution_pool[idx].get_rating(strategy="explore"))
            for idx in candidates
        ]
        explore_scores.sort(key=lambda x: x[1], reverse=True)

        # Take top-k
        selected = [idx for idx, _ in explore_scores[:min(k, len(explore_scores))]]

        # If include_last, ensure the most recent program is included
        # When k=1, don't displace the top UCB pick (matches LUCB behavior where champion is sacred)
        if include_last and len(candidates) > 0 and k > 1:
            last_idx = candidates[-1]
            if last_idx not in selected:
                if len(selected) >= k and len(selected) > 0:
                    selected[-1] = last_idx
                else:
                    selected.append(last_idx)

        # Pad if needed
        last_idx = candidates[-1]
        while len(selected) < k:
            selected.append(last_idx)

        return [self.evolution_pool[i] for i in selected]

    def sample_k_programs_softmax(self, pool_size, k, include_last=True):
        """
        Select k programs by sampling from a softmax distribution over explore scores.
        Temperature is set to the max score for a reasonable spread.

        Args:
            pool_size: Number of recent programs to consider as candidates
            k: Number of programs to select
            include_last: If True, always include the most recent program

        Returns:
            List of Program objects (up to k programs)
        """
        if k <= 0 or len(self.evolution_pool) == 0:
            return []

        indices = list(range(len(self.evolution_pool)))
        candidates = indices[-pool_size:] if pool_size < len(indices) else indices

        if len(candidates) == 0:
            return []

        # Compute explore scores (UCB = mu + 2*sigma)
        scores = np.array([
            self.evolution_pool[idx].get_rating(strategy="explore")
            for idx in candidates
        ])

        # Temperature = max score, floored at 1.0
        temperature = max(float(np.max(scores)), 1.0)

        # Compute softmax probabilities
        shifted = (scores - np.max(scores)) / temperature
        probs = np.exp(shifted) / np.sum(np.exp(shifted))

        # Sample without replacement; cap at len(candidates) to avoid numpy error
        n_sample = min(k, len(candidates))
        sampled_local = np.random.choice(len(candidates), size=n_sample, replace=False, p=probs)
        selected = [candidates[i] for i in sampled_local]

        # If include_last, ensure the most recent program is included (skip when k=1)
        if include_last and len(candidates) > 0 and k > 1:
            last_idx = candidates[-1]
            if last_idx not in selected:
                if len(selected) >= k and len(selected) > 0:
                    selected[-1] = last_idx
                else:
                    selected.append(last_idx)

        # Pad if needed
        last_idx = candidates[-1]
        while len(selected) < k:
            selected.append(last_idx)

        return [self.evolution_pool[i] for i in selected]

    def find_program_with_id(self, program_id):
        found = False
        for program in self.evolution_pool:
            if program.program_id == program_id:
                found = True
                return program
        if not found:
            return None
    
    def get_most_elite_program(self, strategy="explore"):
        assert len(self.evolution_pool) > 0
        if len(self.evolution_pool) == 1:
            return self.evolution_pool[-1]
        
        index_score_pairs = []
        for idx, program in enumerate(self.evolution_pool):
            index_score_pairs.append({"idx": idx, "score": program.get_rating(strategy=strategy)})
        
        index_score_pairs.sort(key=lambda x: x["score"], reverse=True)
        best_idx = index_score_pairs[0]["idx"]
        return self.evolution_pool[best_idx]
    
    def get_latest_program(self):
        assert len(self.evolution_pool) > 0
        return self.evolution_pool[-1]
    
    def state_dict(self):
        state = {}
        state["max_pool_size"] = self.max_pool_size
        state["max_program_id"] = self.max_program_id
        state["evolution_pool"] = [program.state_dict() for program in self.evolution_pool]
        return state
    
    def load_state_dict(self, state):
        self.max_pool_size = state["max_pool_size"]
        self.max_program_id = state["max_program_id"]
        self.evolution_pool = []
        for program_state in state["evolution_pool"]:
            program = Program(
                principles=program_state["principles"],
                program_id=program_state["program_id"]
            )
            program.load_state_dict(program_state)
            self.evolution_pool.append(program)


def main(config: Config):
    # Setup logging
    # ADVICE: Reproducibility.
    # This script uses `random` (evolution sampling, shuffling) and `numpy`/`torch` elsewhere.
    # If you want runs to be more reproducible, consider adding a `seed` field to Config and setting
    # random.seed(seed), np.random.seed(seed), and torch.manual_seed(seed) here.
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project="system_prompt_learning",
        wandb_name=config.experiment_name,
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # set up experiment dir
    experiment_dir = os.path.join("data", config.domain, "train", config.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Load the dataset
    logger.info("Loading dataset...")
    train_dataset_name, test_dataset_name = get_dataset_pair(config.dataset_pair)
    logger.info(f"Using dataset pair: {config.dataset_pair} (train={train_dataset_name}, test={test_dataset_name})")

    train_data_filename = os.path.join(experiment_dir, "train_data.jsonl")
    if os.path.exists(train_data_filename):
        train_data = []
        with open(train_data_filename) as f:
            for line in f:
                train_data.append(json.loads(line))
        print(f"Loaded {len(train_data)} items from train_data.jsonl")
    else:
        train_data = load_data(train_dataset_name)
        if config.dataset_size > 0:
            train_data = train_data[: config.dataset_size]

        with open(train_data_filename, "w") as f:
            for each in train_data:
                f.write(json.dumps(each) + "\n")
        print(f"Wrote {len(train_data)} items to train_data.jsonl")

    # Load test data
    test_data = load_data(test_dataset_name)

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)

    # Reference policy sampling
    ref_model_sample_client = service_client.create_sampling_client(base_model=config.model_name)

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.sampling_temperature,
        top_p=config.top_p,
        stop=renderer.get_stop_sequences(),
    )
    # Optimizer step
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    # start from an existing step
    stats_filename = os.path.join(experiment_dir, "stats.json")
    if os.path.exists(stats_filename):
        stats = json.load(open(stats_filename))
    else:
        stats = {}
    
    trueskill_system = TrueSkillSystem(
        initial_mu=25,
        initial_sigma=25/3,
        beta=25/6,
        tau=25/300,
        draw_probability=0.10
    )
    
    evolution_pool = EvolutionPool(max_pool_size=config.max_pool_size)
    root_program = Program(
        principles={},
        program_id=evolution_pool.get_new_program_id(),
        mu=trueskill_system.initial_mu,
        sigma=trueskill_system.initial_sigma,
        timestep=-1
    )
    evolution_pool.add_program_to_pool(root_program)

    this_test_acc = 0.
    best_test_acc = -1
    best_ckpt_step = -1
    if len(stats) > 0:
        step_count = 0
        for stats_key, stats_val in stats.items():
            if stats_key != f"step_{step_count}":
                continue
            if "test" not in stats_val:
                continue

            if stats_val["test"]["eval_avg_reward"] > best_test_acc:
                best_test_acc = stats_val["test"]["eval_avg_reward"]
                best_ckpt_step = step_count
            step_count += 1
    
    print("best_test_acc:", best_test_acc)
    print("best_ckpt_step:", best_ckpt_step)

    assert config.resume_strategy in ["last", "best"]
    if config.resume_strategy == "last":
        resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    else:
        resume_info = None
        ckpt_info_path = os.path.join(config.log_path, "checkpoints.jsonl")

        if os.path.exists(ckpt_info_path):
            all_checkpoints_info = read_jsonl(ckpt_info_path)
            for ckpt_info in all_checkpoints_info:
                if "step" not in ckpt_info:
                    continue
                if ckpt_info["step"] == best_ckpt_step:
                    resume_info = ckpt_info

    print("Resume info:", resume_info)
    if resume_info:
        print("RESUMING TRAINING RUN ...")
        training_client = service_client.create_training_client_from_state(
            resume_info["state_path"]
        )
        start_step = resume_info["step"]
        logger.info(f"Resuming from {config.log_path} at step {start_step}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_step = 0

    if len(train_data) % config.batch_size == 0:
        num_batches = len(train_data) // config.batch_size
    else:
        num_batches = len(train_data) // config.batch_size + 1

    for epoch in range(config.n_epochs):
        # Init
        print("=" * 30 + f"\nEpoch {epoch}\n" + "=" * 30)
        cur_epoch_dir = os.path.join(experiment_dir, f"epoch_{epoch}")
        os.makedirs(cur_epoch_dir, exist_ok=True)

        # Check if shuffled data already exists for this epoch
        shuffled_filename = os.path.join(cur_epoch_dir, "shuffled_data.jsonl")
        if os.path.exists(shuffled_filename):
            shuffled_data = []
            with open(shuffled_filename) as f:
                for line in f:
                    shuffled_data.append(json.loads(line))
            print(f"Loaded {len(shuffled_data)} items from shuffled_data.jsonl")
        else:
            print(f"Shuffling data ...")
            shuffled_data = copy.deepcopy(train_data)
            random.shuffle(shuffled_data)
            with open(shuffled_filename, "w") as f:
                for each in shuffled_data:
                    f.write(json.dumps(each) + "\n")
            print(f"Wrote {len(shuffled_data)} items to shuffled_data.jsonl")

        #  Main training loop
        for batch_idx in range(num_batches):
            step = epoch * num_batches + batch_idx
            if step < start_step:
                continue

            if f"step_{step}" not in stats:
                stats[f"step_{step}"] = {"epoch": epoch, "batch": batch_idx, "complete": False}
            elif stats[f"step_{step}"]["complete"]:
                print(f"WARNING: step={step} in epoch={epoch} batch_idx={batch_idx} already marked as complete")
                if "rl" in config.train_mode:
                    print(f"Train mode = {config.train_mode}, so weight updates need to be re-computed. Cannot skip this step.")
                    stats[f"step_{step}"] = {"epoch": epoch, "batch": batch_idx, "complete": False}
                else:
                    print(f"Train mode = {config.train_mode}, NO WEIGHT UPDATES NEEDED. Skipping this step.")
                    continue

            # Init
            print(f"Step {step} (Epoch {epoch}, Batch {batch_idx})")
            cur_step_dir = os.path.join(experiment_dir, f"step_{step}")
            os.makedirs(cur_step_dir, exist_ok=True)

            # Get current batch data
            batch_data = copy.deepcopy(shuffled_data[batch_idx * config.batch_size : (batch_idx + 1) * config.batch_size])

            # Setup metrics for logging
            t_start = time.time()
            metrics: dict[str, float] = {
                "progress/batch": batch_idx,
                "optim/lr": config.learning_rate,
                "progress/done_frac": (batch_idx + 1) / num_batches,
            }

            # Set up sampling parameters
            sampling_path = training_client.save_weights_for_sampler(name=f"{step:06d}").result().path
            sampling_client = service_client.create_sampling_client(model_path=sampling_path)

            if step > 0:
                evolution_pool_filename = os.path.join("data", config.domain, "train", config.experiment_name, f"step_{step}/evolution_pool.json")
                print("Loading evolution pool", evolution_pool_filename)
                evolution_state = json.load(open(evolution_pool_filename))
                evolution_pool.load_state_dict(evolution_state)
            
            # Run eval first
            # =============================== TEST BEGINS ===============================
            test_rollout_filename = os.path.join(cur_step_dir, "test_rollout.jsonl")
            # test_time_program = evolution_pool.get_most_elite_program(strategy="explore")
            test_time_program = evolution_pool.get_latest_program()
            stats[f"step_{step}"]["test_time_program_id"] = test_time_program.program_id

            print("Step", step, "test_time prompt ID:", test_time_program.program_id)
            print("     rating", test_time_program.rating)
            print("     past scores:", test_time_program.past_score_history)
            print("     past self-modify:", test_time_program.self_modify_id_list)

            # EVAL: send futures first
            test_time_principles = "\n".join([f"[{i}]. {e}" for i, e in test_time_program.principles.items()])
            test_batch_data = [{
                "prompt": PROBLEM_WITH_PRINCIPLE_TEMPLATE.format(
                    problem=each["problem"],
                    principles=test_time_principles if test_time_principles else "None",
                ),
                **each
            } for each in test_data]

            test_prompts = []
            test_futures = []
            for test_item in test_batch_data:
                convo = [{"role": "user", "content": test_item["prompt"]}]
                model_input = renderer.build_generation_prompt(convo)
                prompt_tokens = model_input.to_ints()
                sample_futures = []
                for _ in range(config.test_group_size):
                    sample_futures.append(
                        sampling_client.sample(
                            prompt=model_input,
                            num_samples=1,
                            sampling_params=sampling_params,
                        )
                    )
                test_prompts.append(prompt_tokens)
                test_futures.append(sample_futures)
            
            # =============================== TEST ENDS ===============================

            # SAMPLING: send futures first
            print("Sampling %d system prompts from most recent %d prompts (strategy=%s)" % (
                config.num_parallel_programs, config.choose_from_most_recent, config.program_selection_strategy))

            if config.program_selection_strategy == "uniform":
                program_lists = evolution_pool.sample_k_programs_uniform(
                    pool_size=config.choose_from_most_recent,
                    k=config.num_parallel_programs,
                    include_last=True,
                )
            elif config.program_selection_strategy == "lucb":
                program_lists = evolution_pool.sample_k_programs_lucb(
                    pool_size=config.choose_from_most_recent,
                    k=config.num_parallel_programs,
                    include_last=True,
                )
            elif config.program_selection_strategy == "ucb":
                program_lists = evolution_pool.sample_k_programs_ucb(
                    pool_size=config.choose_from_most_recent,
                    k=config.num_parallel_programs,
                    include_last=True,
                )
            elif config.program_selection_strategy == "softmax":
                program_lists = evolution_pool.sample_k_programs_softmax(
                    pool_size=config.choose_from_most_recent,
                    k=config.num_parallel_programs,
                    include_last=True,
                )
            else:
                raise ValueError(f"Unknown program_selection_strategy: {config.program_selection_strategy}")
            for program in program_lists:
                print("     - Program ID:", program.program_id)

            batch_data_all_programs = []

            for program_idx in range(config.num_parallel_programs):
                principles = program_lists[program_idx].principles
                # make directory
                cur_program_dir = os.path.join(cur_step_dir, f"sampled_program_{program_idx}")
                os.makedirs(cur_program_dir, exist_ok=True)
                # dump principles
                principle_filename = os.path.join(cur_program_dir, "principles_to_mutate.json")
                json.dump(principles, open(principle_filename, "w"), indent=2)

                # Format the batch data with principles
                formatted_principles_i = "\n".join([f"[{i}]. {e}" for i, e in principles.items()])
                batch_data_for_program = [{
                    "prompt": PROBLEM_WITH_PRINCIPLE_TEMPLATE.format(
                        principles=formatted_principles_i if formatted_principles_i else "None",
                        problem=each["problem"],
                    ) if principles else each["problem"],
                    **each
                } for each in batch_data]

                batch_data_all_programs.append(batch_data_for_program)

            batch_prompts_all_programs = []
            batch_futures_all_programs = []

            batch_tokens_all_programs = []
            batch_ob_lens_all_programs = []
            batch_logprobs_all_programs = []
            batch_completions_all_programs = []
            batch_rewards_all_programs = []

            # Sending the futures
            for batch_data_for_program in batch_data_all_programs:
                batch_prompts_for_program = []
                batch_futures_for_program = []

                for batch_item in batch_data_for_program:
                    convo = [{"role": "user", "content": batch_item["prompt"]}]
                    model_input = renderer.build_generation_prompt(convo)
                    prompt_tokens = model_input.to_ints()

                    # Generate response
                    sample_futures: list[Future[types.SampleResponse]] = []
                    for _ in range(config.group_size):
                        sample_futures.append(
                            sampling_client.sample(
                                prompt=model_input,
                                num_samples=1,
                                sampling_params=sampling_params,
                            )
                        )
                    
                    batch_prompts_for_program.append(prompt_tokens)
                    batch_futures_for_program.append(sample_futures)
                
                batch_prompts_all_programs.append(batch_prompts_for_program)
                batch_futures_all_programs.append(batch_futures_for_program)

            print("Sampling in progress ....")
            sampling_count = 0

            for batch_data_for_program, batch_prompts_for_program, batch_futures_for_program in zip(
                batch_data_all_programs, batch_prompts_all_programs, batch_futures_all_programs
            ):
                batch_tokens_for_program = []
                batch_ob_lens_for_program = []
                batch_logprobs_for_program = []
                batch_completions_for_program = []
                batch_rewards_for_program = []

                for batch_item, prompt_tokens, sample_futures in zip(
                    batch_data_for_program, batch_prompts_for_program, batch_futures_for_program
                ):
                    sampling_count += 1
                    print("     - Sampling count: %d ..." % sampling_count)

                    answer = batch_item["groundtruth"]
                    group_tokens = []
                    group_ob_lens = []
                    group_logprobs = []
                    group_completions = []
                    group_rewards = []

                    for future in sample_futures:
                        sample_result = future.result()
                        sampled_tokens = sample_result.sequences[0].tokens
                        sampled_logprobs = sample_result.sequences[0].logprobs
                        assert sampled_logprobs is not None

                        all_tokens = prompt_tokens + sampled_tokens
                        group_tokens.append(all_tokens)
                        # ADVICE: Logprob/offset alignment.
                        # We set ob_len = len(prompt_tokens) - 1 because (in standard next-token LM training)
                        # the first prompt token has no target/logprob, so prompt contributes (prompt_len-1)
                        # positions in the shifted (input, target) pair.
                        # This assumes `sampled_logprobs` has length == len(sampled_tokens) and corresponds
                        # to the generated tokens only (no extra dummy element). If your sampler returns
                        # logprobs with a leading dummy (like compute_logprobs() often does), this will
                        # silently misalign training. Consider asserting the expected lengths here.
                        group_ob_lens.append(len(prompt_tokens) - 1)
                        group_logprobs.append(sampled_logprobs)

                        parsed_message, _ = renderer.parse_response(sampled_tokens)
                        msg_text = parsed_message["content"]
                        group_completions.append(msg_text)

                        reward = verify_func(msg_text, answer)
                        group_rewards.append(reward)
                    
                    batch_tokens_for_program.append(group_tokens)
                    batch_ob_lens_for_program.append(group_ob_lens)
                    batch_logprobs_for_program.append(group_logprobs)
                    batch_completions_for_program.append(group_completions)
                    batch_rewards_for_program.append(group_rewards)
                
                batch_tokens_all_programs.append(batch_tokens_for_program)
                batch_ob_lens_all_programs.append(batch_ob_lens_for_program)
                batch_logprobs_all_programs.append(batch_logprobs_for_program)
                batch_completions_all_programs.append(batch_completions_for_program)
                batch_rewards_all_programs.append(batch_rewards_for_program)

            print("Using value baseline strategy:", config.value_baseline_computation)
            all_rewards_torch = torch.Tensor(batch_rewards_all_programs)
            assert all_rewards_torch.shape == torch.Size(
                [config.num_parallel_programs, len(batch_data), config.group_size]
            )

            if config.value_baseline_computation == "marginalize":
                print("     - Marginalizing across system prompts ...")
                value_baseline_torch = all_rewards_torch.mean(dim=0, keepdim=True).mean(dim=-1, keepdim=True)
                advantage_torch = all_rewards_torch - value_baseline_torch
            else:
                print("     - Using normal value baseline, for %d system prompts separately ..." % config.num_parallel_programs)
                value_baseline_torch = all_rewards_torch.mean(dim=-1, keepdim=True)
                advantage_torch = all_rewards_torch - value_baseline_torch
            
            batch_advantages_all_programs = advantage_torch.tolist()

            print("Preparing RL Training data ...")
            training_datums = []

            for batch_tokens_for_program, batch_ob_lens_for_program, batch_logprobs_for_program, batch_advantages_for_program in zip(
                batch_tokens_all_programs, batch_ob_lens_all_programs, batch_logprobs_all_programs, batch_advantages_all_programs
            ):
                for group_tokens, group_ob_lens, group_logprobs, advantages in zip(
                    batch_tokens_for_program, batch_ob_lens_for_program, batch_logprobs_for_program, batch_advantages_for_program
                ):
                    # check if all advantages are zero
                    if all(advantage == 0.0 for advantage in advantages):
                        # ADVICE: This can easily drop most data early in training.
                        # With binary rewards (0/1) and small group_size, it's common to get all-0 or all-1 groups,
                        # which makes centered advantages all zero and therefore skips the entire question.
                        # Consider logging the skip ratio and/or using a different baseline/advantage scheme if learning stalls.
                        # Skip question because all advantages are the same
                        continue

                    for tokens, logprob, advantage, ob_len in zip(
                        group_tokens, group_logprobs, advantages, group_ob_lens
                    ):
                        input_tokens = tokens[:-1]
                        input_tokens = [int(token) for token in input_tokens]
                        target_tokens = tokens[1:]
                        all_logprobs = [0.0] * ob_len + logprob
                        all_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)
                        assert (
                            len(input_tokens)
                            == len(target_tokens)
                            == len(all_logprobs)
                            == len(all_advantages)
                        ), (
                            f"len(input_tokens): {len(input_tokens)}, len(target_tokens): {len(target_tokens)}, len(all_logprobs): {len(all_logprobs)}, len(all_advantages): {len(all_advantages)}"
                        )
                        datum = types.Datum(
                            model_input=types.ModelInput.from_ints(tokens=input_tokens),
                            loss_fn_inputs={
                                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                                "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                                "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                            },
                        )
                        training_datums.append(datum)
            
            # EVAL: now collect test evals
            # =============================== TEST BEGINS ===============================
            print("Getting test eval results ...")
            test_completions = []
            test_rewards = []
            for sample_futures, test_item in zip(test_futures, test_batch_data):
                answer = test_item["groundtruth"]
                group_rewards = []
                group_texts = []

                for future in sample_futures:
                    sample_result = future.result()
                    sampled_tokens = sample_result.sequences[0].tokens
                    parsed_message, _ = renderer.parse_response(sampled_tokens)
                    msg_text = parsed_message["content"]
                    group_texts.append(msg_text)
                    reward = verify_func(msg_text, answer)
                    group_rewards.append(reward)
                
                test_completions.append(group_texts)
                test_rewards.append(group_rewards)
            
            this_test_acc = np.mean(test_rewards)
            eval_stats = {"eval_avg_reward": this_test_acc}
            
            eval_correct_counts = torch.Tensor(test_rewards).sum(dim=-1).tolist()
            eval_pass_at_k_list = []
            for test_q_idx in range(len(test_batch_data)):
                q_pass_at_k_list = pass_at_k_for_range(
                    n=config.test_group_size,
                    c=eval_correct_counts[test_q_idx],
                )
                eval_pass_at_k_list.append(q_pass_at_k_list)
            test_eval_pass_at_k = torch.Tensor(eval_pass_at_k_list).mean(dim=0).tolist()
            for k_value in range(1, config.test_group_size + 1):
                eval_stats[f"Pass@{k_value}"] = test_eval_pass_at_k[k_value - 1]
            
            for stats_k, stats_v in eval_stats.items():
                print(f"- Test: {stats_k}: {stats_v}")
            stats[f"step_{step}"]["test"] = eval_stats

            test_rollouts = []
            for test_q_idx, test_sample in enumerate(test_batch_data):
                for test_soln_idx in range(config.test_group_size):
                    runid = test_q_idx * config.test_group_size + test_soln_idx
                    rollout_item = {}
                    rollout_item.update({"runid": runid})
                    rollout_item.update(test_sample)
                    rollout_item.update({
                        "reward": test_rewards[test_q_idx][test_soln_idx],
                        "trajectories": [{
                            "trajectory": [
                                {"role": "user", "content": test_sample["prompt"]},
                                {"role": "assistant", "content": test_completions[test_q_idx][test_soln_idx]}
                            ]
                        }],
                    })
                    test_rollouts.append(rollout_item)
            save_rollouts(test_rollouts, test_rollout_filename)
            # =============================== TEST ENDS ===============================

            # Save checkpoint
            if (step % config.save_every == 0 and step > start_step) or (this_test_acc > best_test_acc):
                if this_test_acc > best_test_acc:
                    best_test_acc = this_test_acc
                    best_ckpt_step = step
                    print("Finding new best test acc:", this_test_acc, "SAVING CHECKPOINT AT STEP:", best_ckpt_step)
                
                if "rl" in config.train_mode:
                    checkpoint_utils.save_checkpoint(
                        training_client=training_client,
                        name=f"{step:06d}",
                        log_path=config.log_path,
                        kind="state",
                        loop_state={"step": step},
                    )

            # TRAINING step
            if "rl" in config.train_mode:
                print("Sending RL step future to server ...")
                if config.rl_loss_fn == "importance_sampling":
                    fwd_bwd_future = training_client.forward_backward(
                        training_datums, loss_fn="importance_sampling"
                    )
                elif config.rl_loss_fn == "cispo":
                    fwd_bwd_future = training_client.forward_backward(
                        training_datums,
                        loss_fn="cispo",
                        loss_fn_config={"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}
                    )
                elif config.rl_loss_fn == "ppo":
                    fwd_bwd_future = training_client.forward_backward(
                        training_datums,
                        loss_fn="ppo",
                    )
                else:
                    raise NotImplementedError
                optim_step_future = training_client.optim_step(adam_params)

            # logging stats
            max_K = config.group_size
            # all_rewards_torch: [num_parallel_programs, num_questions, group_size]
            reshaped_all_rewards = all_rewards_torch.permute(1, 0, 2).flatten(1, 2)
            assert reshaped_all_rewards.shape == torch.Size([len(batch_data), config.num_parallel_programs * config.group_size])
            reshaped_all_rewards = reshaped_all_rewards.tolist()
            rollout_stats = {
                "Agg_avg_reward": all_rewards_torch.mean().item(),
                f"Agg_Pass@{max_K}": sum(max(reward_list) > 0 for reward_list in reshaped_all_rewards) / len(reshaped_all_rewards)
            }

            all_correct_counts = all_rewards_torch.sum(dim=-1).tolist()
            all_pass_at_k_list = []
            for program_index in range(config.num_parallel_programs):
                program_pass_at_k_list = []
                for problem_index in range(len(batch_data)):
                    problem_pass_at_k_list = pass_at_k_for_range(
                        n=config.group_size,
                        c=all_correct_counts[program_index][problem_index],
                    )
                    program_pass_at_k_list.append(problem_pass_at_k_list)
                all_pass_at_k_list.append(program_pass_at_k_list)
            all_pass_at_k_tensor = torch.Tensor(all_pass_at_k_list)
            assert all_pass_at_k_tensor.shape == torch.Size([config.num_parallel_programs, len(batch_data), config.group_size])

            problem_pass_at_k_tensor = all_pass_at_k_tensor.mean(dim=0)
            assert problem_pass_at_k_tensor.shape == torch.Size([len(batch_data), config.group_size])
            rollout_stats.update({
                "avg_question_Pass@1": torch.mean(problem_pass_at_k_tensor[:, 0]).item(),
                "min_question_Pass@1": torch.min(problem_pass_at_k_tensor[:, 0]).item(),
                "max_question_Pass@1": torch.max(problem_pass_at_k_tensor[:, 0]).item(),
                f"avg_question_Pass@{max_K}": torch.mean(problem_pass_at_k_tensor[:, -1]).item(),
                f"min_question_Pass@{max_K}": torch.min(problem_pass_at_k_tensor[:, -1]).item(),
                f"max_question_Pass@{max_K}": torch.max(problem_pass_at_k_tensor[:, -1]).item(),
            })

            program_pass_at_k_tensor = all_pass_at_k_tensor.mean(dim=1)
            assert program_pass_at_k_tensor.shape == torch.Size([config.num_parallel_programs, config.group_size])
            rollout_stats.update({
                "avg_PROGRAM_Pass@1": torch.mean(program_pass_at_k_tensor[:, 0]).item(),
                "min_PROGRAM_Pass@1": torch.min(program_pass_at_k_tensor[:, 0]).item(),
                "max_PROGRAM_Pass@1": torch.max(program_pass_at_k_tensor[:, 0]).item(),
                f"avg_PROGRAM_Pass@{max_K}": torch.mean(program_pass_at_k_tensor[:, -1]).item(),
                f"min_PROGRAM_Pass@{max_K}": torch.min(program_pass_at_k_tensor[:, -1]).item(),
                f"max_PROGRAM_Pass@{max_K}": torch.max(program_pass_at_k_tensor[:, -1]).item(),
            })

            for stats_k, stats_v in rollout_stats.items():
                print(f"- {stats_k}: {stats_v}")
            stats[f"step_{step}"]["rollout"] = rollout_stats

            # Double check pass@1 accuracy
            program_accuracies = all_rewards_torch.mean(dim=-1).mean(dim=-1).tolist()
            avg_program_acc = np.mean(program_accuracies)
            avg_program_pass_1 = torch.mean(program_pass_at_k_tensor[:, 0]).item()
            assert np.abs(avg_program_acc - avg_program_pass_1) < 0.01

            # Running evolutionary procedure
            # start with the simplest score: pass@1
            program_scores = program_pass_at_k_tensor[:, 0].tolist()
            # update program scores in Program instances
            for program, program_score in zip(program_lists, program_scores):
                program.add_eval_score(program_score)
            
            # Need sorted rating list (1st place, 2nd place, ...) to update the ratings
            sorted_program_indices = sorted(range(len(program_scores)), key=lambda p_i: program_scores[p_i], reverse=True)
            # edge case: there might be repeat program_id's, so we implement a unique() function
            sorted_program_indices_unique = []
            seen_program_ids = set()
            for p_idx in sorted_program_indices:
                program_id = program_lists[p_idx].program_id
                if program_id not in seen_program_ids:
                    seen_program_ids.add(program_id)
                    sorted_program_indices_unique.append(p_idx)
            ties = scores_to_ties([program_scores[p_i] for p_i in sorted_program_indices_unique], tolerance=0.01)

            best_program_index = sorted_program_indices_unique[0]
            print("Selecting best program from scores:", program_scores)
            print("     Prior ratings:", [program.rating for program in program_lists])
            print("     best_program_index:", best_program_index, "program_id:", program_lists[best_program_index].program_id, "rating:", program_lists[best_program_index].rating)
            print("     Sorted indices:", sorted_program_indices, " ====> ", sorted_program_indices_unique, " ; ties:", ties)

            updated_ratings = trueskill_system.rate_ranking(
                [program_lists[p_i].rating for p_i in sorted_program_indices_unique],
                ties=ties, apply_dynamics=True
            )
            for p_i, updated_rating in zip(sorted_program_indices_unique, updated_ratings):
                program_lists[p_i].update_rating(updated_rating)
            
            print("     Updated ratings:", [program.rating for program in program_lists])
            print("      - best_program_index updated rating:", program_lists[best_program_index].rating, "\n")

            # Save rollouts
            rollouts_all_programs = []
            for program_idx, batch_data_for_program in enumerate(batch_data_all_programs):
                rollouts = []
                cur_program_dir = os.path.join(cur_step_dir, f"sampled_program_{program_idx}")
                assert os.path.exists(cur_program_dir)
                rollout_filename = os.path.join(cur_program_dir, "rollout.jsonl")

                for question_idx, sample in enumerate(batch_data_for_program):
                    for solution_idx in range(config.group_size):
                        runid = question_idx * config.group_size + solution_idx
                        rollout_item = {}
                        rollout_item.update({"runid": runid})
                        rollout_item.update(sample)
                        rollout_item.update({
                            "reward": batch_rewards_all_programs[program_idx][question_idx][solution_idx],
                            "trajectories": [{
                                "trajectory": [
                                    {"role": "user", "content": sample["prompt"]},
                                    {"role": "assistant", "content": batch_completions_all_programs[program_idx][question_idx][solution_idx]}
                                ]
                            }],
                        })
                        rollouts.append(rollout_item)

                rollouts_all_programs.append(rollouts)
                save_rollouts(rollouts, rollout_filename)
            
            # Run Mutation + Crossover
            if "evolution" in config.train_mode:
                assert config.evolution_sample_client in ["reference", "current"]
                if config.evolution_sample_client == "reference":
                    print("(Using REFERENCE policy for evolution)")
                    evolution_sampling_server = ref_model_sample_client
                else:
                    print("(Using CURRENT policy for evolution)")
                    evolution_sampling_server = sampling_client
                
                selected_principles = program_lists[best_program_index].principles
                selected_rollouts = rollouts_all_programs[best_program_index]

                # Generate critiques and update principles
                next_principle_filename = os.path.join(cur_step_dir, "evolved_principles.json")
                if os.path.exists(next_principle_filename):
                    print("Overwriting:", next_principle_filename)

                crossover_future = None
                if random.random() < config.crossover_prob and len(sorted_program_indices_unique) >= 2:
                    print("Starting Crossover ...")
                    
                    # Compute per-problem scores for each program [num_parallel_programs, num_questions]
                    per_problem_scores = all_rewards_torch.mean(dim=-1)
                    # For each problem, find which unique program won
                    wins_by_program = {prog_idx: [] for prog_idx in sorted_program_indices_unique}

                    tolerance = 0.01
                    for problem_idx in range(len(batch_data)):
                        scores = {prog_idx: per_problem_scores[prog_idx, problem_idx].item() for prog_idx in sorted_program_indices_unique}
                        max_score = max(scores.values())
                        winners = [prog_idx for prog_idx, score in scores.items() if abs(score - max_score) < tolerance]
                        if len(winners) == 1:
                            winner_prog_idx = winners[0]
                        else:
                            continue
                        problem_text = batch_data[problem_idx]["problem"]
                        wins_by_program[winner_prog_idx].append({problem_idx: problem_text})
                    
                    # Keep Set A + any program that won at least one problem
                    programs_to_include = [best_program_index]
                    for prog_idx in sorted_program_indices_unique[1:]:
                        if wins_by_program[prog_idx]:
                            programs_to_include.append(prog_idx)
                    
                    if len(programs_to_include) >= 2 and len(wins_by_program[best_program_index]) > 0:
                        system_prompt_list = [program_lists[prog_idx].principles for prog_idx in programs_to_include]
                        problems_each_prompt_did_best = [wins_by_program[prog_idx] for prog_idx in programs_to_include]
                        crossover_prompt = build_crossover_prompt(system_prompt_list, problems_each_prompt_did_best)

                        with open(os.path.join(cur_step_dir, "crossover_prompt.txt"), "w") as file:
                            file.write(crossover_prompt)
                        
                        crossover_future = get_sample_future(
                            [{"role": "user", "content": crossover_prompt}], renderer, evolution_sampling_server, sampling_params
                        )
                        crossover_parent_ids = [program_lists[idx].program_id for idx in programs_to_include]
                        crossover_mu, crossover_sigma = calculate_crossover_rating(
                            [program_lists[idx].rating for idx in programs_to_include], crossover_sigma=config.crossover_sigma
                        )
                        print("Sent Crossover sampling future ...")
                    else:
                        print("Skipping Crossover ...")

                # Self reflection
                print("Self reflection begins ...")
                new_principles = None
                try:
                    principle_updater = PrincipleUpdater(
                        renderer, 
                        evolution_sampling_server,
                        sampling_params=sampling_params,
                        max_operations=config.max_update_operations,
                    )
                    # use selected_rollouts and selected_principles
                    new_principles = principle_updater.run(
                        rollouts=selected_rollouts, 
                        principles=selected_principles,
                        save_dir=cur_step_dir,
                    )
                    json.dump(new_principles, open(next_principle_filename, "w"), indent=2)
                    print(f"Saved {len(new_principles)} principles to {next_principle_filename}")
                except Exception as mutation_e:
                    print(f"Encountered error: {mutation_e}")

                if crossover_future:
                    crossover_response = get_text_from_sampled_future(crossover_future, renderer)
                    crossover_response_filepath = os.path.join(cur_step_dir, "crossover_response.txt")
                    with open(crossover_response_filepath, "w") as file:
                        file.write(crossover_response)
                    
                    crossover_operations = get_operations_from_json(crossover_response)
                    # Apply crossover operations to create new principles (starting from Set A)
                    if len(crossover_operations) > 0:
                        prev_best_principles = copy.deepcopy(program_lists[best_program_index].principles)
                        crossover_principles = process_operations(operations=crossover_operations, principles_to_modify=prev_best_principles)
                        # Save crossover results
                        crossover_filename = os.path.join(cur_step_dir, "crossover_result.json")
                        json.dump(crossover_principles, open(crossover_filename, "w"), indent=2)
                        print(f"Saved {len(crossover_principles)} principles to {crossover_filename}")
                        crossover_program = Program(
                            principles=crossover_principles,
                            program_id=evolution_pool.get_new_program_id(),
                            parents_list=crossover_parent_ids,
                            mu=crossover_mu,
                            sigma=crossover_sigma,
                            timestep=step
                        )
                        evolution_pool.add_program_to_pool(crossover_program)
                        # update children list in parent programs
                        for idx in programs_to_include:
                            program_lists[idx].add_child(crossover_program.program_id)
                        print(f"Crossover complete: created program {crossover_program.program_id} from parents {crossover_parent_ids}")

                # add new_principles to evolution pool
                if new_principles:
                    mutation_mu, mutation_sigma = calculate_mutation_rating(program_lists[best_program_index].rating, mutation_sigma=config.mutation_sigma)
                    new_program = Program(
                        principles=new_principles,
                        program_id=evolution_pool.get_new_program_id(),
                        self_modified_from=program_lists[best_program_index].program_id,
                        mu=mutation_mu,
                        sigma=mutation_sigma,
                        timestep=step
                    )
                    evolution_pool.add_program_to_pool(new_program)
                    # update self_modify list in Program instances
                    program_lists[best_program_index].add_self_modify(new_program.program_id)

                    print(f"Mutation complete: created program {new_program.program_id} from parents {program_lists[best_program_index].program_id}")

            # save new evolution pool
            evolution_dir_path = os.path.join("data", config.domain, "train", config.experiment_name, f"step_{step+1}")
            os.makedirs(evolution_dir_path, exist_ok=True)
            evolution_pool_filename = os.path.join(evolution_dir_path, "evolution_pool.json")
            evolution_state = evolution_pool.state_dict()
            with open(evolution_pool_filename, "w") as evo_file:
                json.dump(evolution_state, evo_file, indent=2)
            
            # Get weight updates
            if "rl" in config.train_mode:
                print("Weight update happening ...")
                _fwd_bwd_result = fwd_bwd_future.result()
                _optim_result = optim_step_future.result()

            # Save stats
            stats[f"step_{step}"]["complete"] = True
            json.dump(stats, open(stats_filename, "w"), indent=2)

            # Log metrics[]
            metrics["time/total"] = time.time() - t_start
            metrics.update({f"rollout/{_key}": _val for _key, _val in rollout_stats.items()})
            metrics.update({f"testing/{_key}": _val for _key, _val in eval_stats.items()})
            ml_logger.log_metrics(metrics, step=batch_idx)

    # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"step": step},
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
