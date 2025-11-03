"""
This module contains task configurations and prompt functions for evaluating
LLM models on Kyrgyz datasets.
Each task is defined using the `LightevalTaskConfig` class with its respective
prompt function.
MMLU is separated by subject and also all in one.
"""

from enum import Enum
from typing import List, Optional

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
INTEGER_INDICES = list(map(str, list(range(1, 27))))

class HFSubsets(Enum):
    """Enum for all available Hugging Face dataset subsets in Kyrgyz evaluation tasks."""


    # ============== MMLU
    HF_BASE_MMLU_REPO = "TTimur/kyrgyzMMLU"
    HF_REVISION = None # "209c5b5f999cae5c02eef5735eb817ead18ac214"

    # MMLU (All-inclusive Task Entry)
    MMLU_KYRGYZ_ALL = "kyrgyz_mmlu_all"

    # MMLU by subject
    KYRGYZ_MMLU_HISTORY = "kyrgyz_mmlu_history"
    KYRGYZ_MMLU_LITERATURE = "kyrgyz_mmlu_literature"
    KYRGYZ_MMLU_MEDICINE = "kyrgyz_mmlu_medicine"
    KYRGYZ_MMLU_LANG = "kyrgyz_mmlu_lang"
    KYRGYZ_MMLU_BIOLOGY = "kyrgyz_mmlu_biology"
    KYRGYZ_MMLU_CHEMISTRY = "kyrgyz_mmlu_chemistry"
    KYRGYZ_MMLU_GEOGRAPHY = "kyrgyz_mmlu_geography"
    KYRGYZ_MMLU_MATH = "kyrgyz_mmlu_math"
    KYRGYZ_MMLU_PHYSICS = "kyrgyz_mmlu_physics"
    

    # ============== Reading Comprehension (All-inclusive Task Entry)
    HF_BASE_RC_REPO = "TTimur/kyrgyzRC"
    KYRGYZ_RC_ALL = "kyrgyz_rc_all"

    # Reading Comprehension by subject
    KYRGYZ_RC_LITERATURE = "kyrgyz_rc_literature"
    KYRGYZ_RC_MATH = "kyrgyz_rc_math"
    KYRGYZ_RC_NEWS = "kyrgyz_rc_news"
    KYRGYZ_RC_WIKI = "kyrgyz_rc_wiki"


    # ============== Hellaswag
    HF_BASE_HELLASWAG_REPO = "TTimur/hellaswag_kg"
    


    # ============== Winogrande
    HF_BASE_WINOGRANDE_REPO = "TTimur/winogrande_kg"
    

    # ============== TruthfulQA
    HF_BASE_TRUTHFULQA_REPO = "TTimur/truthfulqa_kg"
    

    # ============== GSM8K
    HF_BASE_GSM8K_REPO = "TTimur/gsm8k_kg"
    
    # ============== BoolQ
    HF_BASE_BOOLQ_REPO = "TTimur/boolq_kg"
    

    
# MMLU and RC prompt
def kyrgyz_eval_prompt(line: dict, task_name: Optional[str] = None) -> Doc:
    """
    Creates a prompt for a multiple-choice task in Kyrgyz for MMLU and RC. This function formats the prompt
    based on the provided query and choices, handling both standard tasks and MMLU-specific
    tasks (if "mmlu" is part of the task name).

    The prompt includes an instruction in Kyrgyz, followed by the query, available choices,
    and finally the correct answer. The function determines how to compute the correct answer
    based on whether the task name contains "mmlu".

    Args:
        line (dict): A dictionary containing the following keys:
            - "query" (str): The question or query to present to the user.
            - "choices" (list of str): A list of possible answer choices.
            - "answer" (int or str): The correct answer, either as an index (for regular tasks)
               or as a string (for MMLU tasks).
        task_name (Optional[str]): The name of the task. If "mmlu" is in the task name, the
            function treats the task as an MMLU task and searches for the correct answer
            by matching the string value of the answer.

    Returns:
        Doc: A `Doc` object containing the formatted prompt, choices, and the correct answer index.
        The `Doc` object includes the following fields:
            - task_name (Optional[str]): The name of the task.
            - query (str): The formatted query prompt in Serbian, including instructions and choices.
            - choices (list of str): The list of available answer choices.
            - gold_index (int): The index of the correct answer.
            - instruction (str): The instruction shown to the user in Serbian.
    """

    question = line["Суроо (KG)"]
    correct_answer = str(line["Туура жооп"])

    if task_name and "mmlu" in task_name:
        choices = [line['А (KG)'], line['Б (KG)'], line['В (KG)'], line['Г (KG)'], line['Д (KG)']]
        choices = [c.strip() for c in choices if c]

        letter_to_index = {
            'а': 0,
            'б': 1,
            'в': 2,
            'г': 3,
            'д': 4
        }
        gold_index = letter_to_index.get(correct_answer.lower(), None)
        
        instruction = """
        Сиз билимиңизге жана жөндөмүңүзгө жараша суроолорго жооп берген AIсыз.\n
        Сизге суроо жана 2-5 жооп варианты берилет, туура жооптун НОМЕРИН (индексин) гана кайтарышыңыз керек.\n
        """

        # Build the query and determine the gold_index in a single pass
        query = f"{instruction}Суроо: {question}\n\nСунушталган жооптор:\n"
        

        # Show all choises
        for i, choice in enumerate(choices):
            if choice:
                query += f"{i}. {choice}\n"

        query += "\n\nТуура жоопту тандаңыз: "

    # Reading Comprehension prompt
    else:
        choices = [line['А (KG)'], line['Б (KG)'], line['В (KG)'], line['Г (KG)']]
        choices = [c.strip() for c in choices if c]

        letter_to_index = {
            'а': 0,
            'б': 1,
            'в': 2,
            'г': 3,
            'д': 4
        }
        gold_index = letter_to_index.get(correct_answer.lower(), None)

        text = line['Текст (KG)']
        
        instruction = """
        Сизге бир темага байланыштуу бир нече үзүндү текст берилген. Бардык үзүндүлөрдү кунт коюп окуп, андан кийин төмөндөгү суроолорго жооп бериңиздер.\n
        Суроо менен 2-4 жооп варианты берилет, туура жооптун НОМЕРИН (индексин) гана кайтарышыңыз керек.\n
        """

        # Build the query and determine the gold_index in a single pass
        query = f"{instruction}Текст: {text}\n\nСуроо: {question}\n\nСунушталган жооптор:\n"
        

        # Show all choises
        for i, choice in enumerate(choices):
            if choice:
                query += f"{i}. {choice}\n"

        query += "\n\nТуура жоопту тандаңыз: "
    

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=instruction,
    )


# TODO: check if this is correct
# HELLASWAG prompt
def kyrgyz_hellaswag_prompt(line, task_name: str = None):

    ctx_a_kg = line['ctx_a_kg'] if line['ctx_a_kg'] else '.'
    ctx_b_kg = line['ctx_b_kg'].capitalize() if line['ctx_b_kg'] else '.'

    query = "Төмөндө жалпы түшүнүккө (common sense) байланыштуу бир нече тандоо суроолору (жооптору менен) берилген.\n\n"
    query += f"Суроо: {line['activity_label_kg']}: {ctx_a_kg} {ctx_b_kg}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["endings_kg"])])
    query += "Туура жоопту тандаңыз: "

    gold_ix = int(line["label"]) if line["label"] != "" else -1
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + i for i in LETTER_INDICES[: len(line["endings_kg"])]],
        gold_index=gold_ix,  # -1 for test,
        instruction="Төмөндө жалпы түшүнүккө (common sense) байланыштуу бир нече тандоо суроолору (жооптору менен) берилген.\n\n",
    )

# TODO: check if this is correct
# Winogrande prompt
def kyrgyz_winogrande_prompt(line, task_name: str = None):
    # LL of query + choices
    query, end_of_target = line["sentence_kg"].split("_")
    end_of_target = end_of_target.strip()
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f"{line['option1_kg']} {end_of_target}", f"{line['option2_kg']} {end_of_target}"],
        gold_index=int(line["answer"]) - 1 if line["answer"] != "" else -1,  # managing unk test index
        # "metric": "choices_loglikelihood",
    )


# TODO: check if this is correct
# Truthful_qa prompt
def kyrgyz_truthful_qa_multiple_choice(line, task_name: str = None):
    import ast
    
    # Parse the stringified dictionaries
    mc1 = line.get("mc1_targets_kg", "{}")
    mc2 = line.get("mc2_targets_kg", "{}")
    
    # Convert string representations to actual dictionaries
    if isinstance(mc1, str):
        try:
            mc1 = ast.literal_eval(mc1)
        except (ValueError, SyntaxError):
            mc1 = {"choices": [], "labels": []}
    else:
        mc1 = {"choices": [], "labels": []}
        
    if isinstance(mc2, str):
        try:
            mc2 = ast.literal_eval(mc2)
        except (ValueError, SyntaxError):
            mc2 = {"choices": [], "labels": []}
    else:
        mc2 = {"choices": [], "labels": []}

    choices = [f" {c}" for c in mc1.get("choices", [])] + [f" {c}" for c in mc2.get("choices", [])]
    labels = mc1.get("labels", []) + mc2.get("labels", [])

    return Doc(
        task_name=task_name,
        query=f"Суроо: {line['Question_kg']}\nЖооп:",
        choices=choices,
        gold_index=[ix for ix, label in enumerate(labels) if label == 1],
        specific={"len_mc1": len(mc1.get("choices", []))},
    )


# def truthful_qa_generative(line, task_name: str = None):  # BLEU and combination of BLEU
#     correct_answers = [
#         answer.strip() + "" if answer[-1] == "." else "." for answer in line["correct_answers"] if answer != ""
#     ]
#     if "I have no comment." not in correct_answers:
#         correct_answers.append("I have no comment.")
#     incorrect_answers = [
#         answer.strip() + "" if answer[-1] == "." else "." for answer in line["incorrect_answers"] if answer != ""
#     ]

#     return Doc(
#         task_name=task_name,
#         query=line["question"].strip(),
#         choices=correct_answers + incorrect_answers,
#         gold_index=list(range(len(correct_answers))),
#         specific={"len_mc1": len(line["mc1_targets"]["choices"])},
#     )

# TODO: check if this is correct
# GSM8k prompt
def kyrgyz_gsm8k_prompt(line, task_name: str = None):
    # Has special analysis in metric for number decomposition
    return Doc(
        task_name=task_name,
        query=f"Суроо: {line['question_kg']}\nЖооп:",
        choices=[f" {line['answer_kg']}"],
        gold_index=0,
    )

# TODO: check if this is correct
# BoolQ prompt
def kyrgyz_boolq_prompt(line, task_name: str = None):
    # remove extra `?`
    question = line["question_kg"][:-1] if line["question_kg"][-2:] == "??" else line["question_kg"]
    return Doc(
        task_name=task_name,
        query=f"Текст: {line['passage_kg']}\nСуроо: {question}\nЖооп:",
        choices=[" ооба", " жок"],
        gold_index=["ооба", "жок"].index(line["answer_kg"]),
    )


def create_task_config(
    task_name: str,
    prompt_function,
    hf_repo: str,
    hf_subset: str,
    metric: List,
    evaluation_splits: List[str] = ["test"],
    suite: List[str] = ["community"],
    hf_avail_splits: List[str] = ["test", "validation"],
    few_shots_split: str = "validation",
    generation_size=5,
    few_shots_select="sequential",
) -> LightevalTaskConfig:
    """
    Creates a task configuration using dependency injection for flexible task creation.

    Args:
        task_name: The name of the task.
        prompt_function: The function to generate task prompts.
        hf_repo: Hugging Face repository.
        hf_subset: Subset of the dataset.
        metric: The metric(s) to use for the task.
        evaluation_splits: The evaluation splits to use (default is "test").
        suite: The suite of tasks.
        hf_avail_splits: Available splits (default is "test", "validation").
        few_shots_split: Split used for few-shot examples.

    Returns:
        A `LightevalTaskConfig` object for the task configuration.
    """
    return LightevalTaskConfig(
        name=task_name,
        prompt_function=prompt_function,
        suite=suite,
        hf_repo=hf_repo,
        hf_subset=hf_subset,
        hf_avail_splits=hf_avail_splits,
        evaluation_splits=evaluation_splits,
        few_shots_split=few_shots_split,
        few_shots_select="sequential",
        metric=metric,
        generation_size=generation_size,
        # Since we use trust_dataset, we have to be careful about what is inside the dataset
        # script. We thus lock the revision to ensure that the script doesn't change
        hf_revision=HFSubsets.HF_REVISION.value,
        trust_dataset=True,
        version=0,
    )




# ============================================
# ====== MMLU (All-inclusive Task Entry) =====
# ============================================

mmlu_all = create_task_config(
    task_name="kyrgyz_evals:kyrgyz_mmlu_all",
    prompt_function=kyrgyz_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_MMLU_REPO.value,
    hf_subset=HFSubsets.MMLU_KYRGYZ_ALL.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

# ============================================
# ======= kyrgyz MMLU bench            =======
# ============================================

mmlu_history = create_task_config(
    task_name="kyrgyz_evals:kyrgyz_mmlu_history",
    prompt_function=kyrgyz_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_MMLU_REPO.value,
    hf_subset=HFSubsets.KYRGYZ_MMLU_HISTORY.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_literature = create_task_config(
    task_name="kyrgyz_evals:kyrgyz_mmlu_literature",
    prompt_function=kyrgyz_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_MMLU_REPO.value,
    hf_subset=HFSubsets.KYRGYZ_MMLU_LITERATURE.value,
    metric=[Metrics.loglikelihood_acc_norm],
)


mmlu_college_medicine = create_task_config(
    task_name="kyrgyz_evals:kyrgyz_mmlu_medicine",
    prompt_function=kyrgyz_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_MMLU_REPO.value,
    hf_subset=HFSubsets.KYRGYZ_MMLU_MEDICINE.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_kyrgyz_language = create_task_config(
    task_name="kyrgyz_evals:kyrgyz_mmlu_lang",
    prompt_function=kyrgyz_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_MMLU_REPO.value,
    hf_subset=HFSubsets.KYRGYZ_MMLU_LANG.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_biology = create_task_config(
    task_name="kyrgyz_evals:kyrgyz_mmlu_biology",
    prompt_function=kyrgyz_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_MMLU_REPO.value,
    hf_subset=HFSubsets.KYRGYZ_MMLU_BIOLOGY.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_chemistry = create_task_config(
    task_name="kyrgyz_evals:kyrgyz_mmlu_chemistry",
    prompt_function=kyrgyz_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_MMLU_REPO.value,
    hf_subset=HFSubsets.KYRGYZ_MMLU_CHEMISTRY.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_geography = create_task_config(
    task_name="kyrgyz_evals:kyrgyz_mmlu_geography",
    prompt_function=kyrgyz_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_MMLU_REPO.value,
    hf_subset=HFSubsets.KYRGYZ_MMLU_GEOGRAPHY.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_math = create_task_config(
    task_name="kyrgyz_evals:kyrgyz_mmlu_math",
    prompt_function=kyrgyz_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_MMLU_REPO.value,
    hf_subset=HFSubsets.KYRGYZ_MMLU_MATH.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_physics = create_task_config(
    task_name="kyrgyz_evals:kyrgyz_mmlu_physics",
    prompt_function=kyrgyz_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_MMLU_REPO.value,
    hf_subset=HFSubsets.KYRGYZ_MMLU_PHYSICS.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

# ===================================================
# =======  Reading Comprehension ALL bench ==========
# ===================================================

rc_all = create_task_config(
    task_name="kyrgyz_evals:kyrgyz_rc_all",
    prompt_function=kyrgyz_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_RC_REPO.value,
    hf_subset=HFSubsets.KYRGYZ_RC_ALL.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

# ===================================================
# ======= kyrgyz Reading Comprehension bench ========
# ===================================================

rc_literature = create_task_config(
    task_name="kyrgyz_evals:kyrgyz_rc_literature",
    prompt_function=kyrgyz_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_RC_REPO.value,
    hf_subset=HFSubsets.KYRGYZ_RC_LITERATURE.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

rc_math = create_task_config(
    task_name="kyrgyz_evals:kyrgyz_rc_math",
    prompt_function=kyrgyz_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_RC_REPO.value,
    hf_subset=HFSubsets.KYRGYZ_RC_MATH.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

rc_news = create_task_config(
    task_name="kyrgyz_evals:kyrgyz_rc_news",
    prompt_function=kyrgyz_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_RC_REPO.value,
    hf_subset=HFSubsets.KYRGYZ_RC_NEWS.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

rc_wiki = create_task_config(
    task_name="kyrgyz_evals:kyrgyz_rc_wiki",
    prompt_function=kyrgyz_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_RC_REPO.value,
    hf_subset=HFSubsets.KYRGYZ_RC_WIKI.value,
    metric=[Metrics.loglikelihood_acc_norm],
)


# ===================================================
# ======= kyrgyz hellaswag bench ==================
# ===================================================

hellaswag_kg = create_task_config(
    task_name="kyrgyz_evals:hellaswag_kg",
    prompt_function=kyrgyz_hellaswag_prompt,
    hf_repo=HFSubsets.HF_BASE_HELLASWAG_REPO.value,
    hf_subset='default',
    evaluation_splits=["validation"],
    hf_avail_splits=["train", "validation"],
    few_shots_split="train",
    metric=[Metrics.loglikelihood_acc_norm],
)


# ===================================================
# ======= kyrgyz winogrande bench ==================
# ===================================================

winogrande_kg = create_task_config(
    task_name="kyrgyz_evals:winogrande_kg",
    prompt_function=kyrgyz_winogrande_prompt,
    hf_repo=HFSubsets.HF_BASE_WINOGRANDE_REPO.value,
    hf_subset='default',
    evaluation_splits=["dev"],
    hf_avail_splits=["train", "dev"],
    few_shots_split="train",
    metric=[Metrics.loglikelihood_acc_norm],
)


# ===================================================
# ======= kyrgyz TruthfulQA bench ===================    
# ===================================================

truthfulqa_kg = create_task_config(
    task_name="kyrgyz_evals:truthfulqa_mc_kg",
    prompt_function=kyrgyz_truthful_qa_multiple_choice,
    hf_repo=HFSubsets.HF_BASE_TRUTHFULQA_REPO.value,
    hf_subset='default',
    evaluation_splits=["test"],
    hf_avail_splits=["test", "validation"],
    few_shots_split="validation",
    metric=[Metrics.truthfulqa_mc_metrics], 
)


# ===================================================
# ======= kyrgyz gsm8k_kg bench =====================
# ===================================================

gsm8k_kg = create_task_config(
    task_name="kyrgyz_evals:gsm8k_kg",
    prompt_function=kyrgyz_gsm8k_prompt,
    hf_repo=HFSubsets.HF_BASE_GSM8K_REPO.value,
    hf_subset='default',
    evaluation_splits=["test"],
    hf_avail_splits=["train", "test"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=256,
    metric=[Metrics.quasi_exact_match_gsm8k]
)


# ===================================================
# ======= kyrgyz boolq_kg bench =====================
# ===================================================

boolq_kg = create_task_config(
    task_name="kyrgyz_evals:boolq_kg",
    prompt_function=kyrgyz_boolq_prompt,
    hf_repo=HFSubsets.HF_BASE_BOOLQ_REPO.value,
    hf_subset='default',
    evaluation_splits=["val"],
    hf_avail_splits=["train", "val"],
    few_shots_split="train",
    metric=[Metrics.loglikelihood_acc_norm], 
)




TASKS_TABLE = [
    mmlu_all,
    mmlu_history,
    mmlu_literature,
    mmlu_college_medicine,
    mmlu_kyrgyz_language,
    mmlu_biology,
    mmlu_chemistry,
    mmlu_geography,
    mmlu_math,
    mmlu_physics,
    rc_all,
    rc_literature,
    rc_math,
    rc_news,
    rc_wiki,
    hellaswag_kg,
    winogrande_kg,
    truthfulqa_kg,
    gsm8k_kg,
    boolq_kg
]