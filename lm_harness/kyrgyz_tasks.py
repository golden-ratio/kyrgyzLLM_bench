import numpy as np
from typing import List, Dict, Any
from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_task


# --- Base class for all Kyrgyz language evaluation tasks ---
class KyrgyzBaseTask(ConfigurableTask):
    """
    A base class for Kyrgyz evaluation tasks.
    It handles dataset loading, document processing, and results aggregation.
    """
    VERSION = 1
    DATASET_PATH = "TTimur/kyrgyz-llm-benchmark"  # Main HF dataset path
    OUTPUT_TYPE = "loglikelihood"
    DATASET_NAME = None

    def __init__(self, config):
        clean_config = config.copy()
        clean_config.pop("python_file", None)
        clean_config.pop("class", None)
        if "repeats" not in clean_config or clean_config["repeats"] is None:
            clean_config["repeats"] = 1
        super().__init__(config=clean_config)
        self.repeats = clean_config["repeats"]

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        # Loads a specific configuration (e.g., 'kyrgyz_mmlu_history') from the main dataset
        assert self.DATASET_NAME is not None, "DATASET_NAME must be defined in a subclass"
        from datasets import load_dataset
        self.dataset = load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )

    # --- Document split configuration: 'test' for evaluation, 'validation' for few-shot examples ---
    def has_training_docs(self) -> bool: return False
    def has_validation_docs(self) -> bool: return True
    def has_test_docs(self) -> bool: return True
    def training_docs(self): return []
    def validation_docs(self): return self.dataset["validation"]
    def test_docs(self): return self.dataset["test"]

    def _get_choices(self, doc: Dict[str, Any]) -> List[str]:
        # Extracts choices and prepends a space for loglikelihood calculation
        choices = []
        for choice_key in ["А (KG)", "Б (KG)", "В (KG)", "Г (KG)", "Д (KG)"]:
            if choice_key in doc and doc[choice_key] and str(doc[choice_key]).strip():
                choices.append(" " + str(doc[choice_key]).strip())
        return choices

    def _get_correct_answer_index(self, doc: Dict[str, Any], choices: List[str]) -> int:
        # Converts the correct letter (e.g., "в") to a zero-based index (e.g., 2)
        correct_letter = str(doc["Туура жооп"]).strip().lower()
        letter_to_index = { "а": 0, "б": 1, "в": 2, "г": 3, "д": 4 }
        return letter_to_index.get(correct_letter, -1)

    def process_results(self, doc: Dict[str, Any], results: List[float]) -> Dict[str, float]:
        if not results:
            return {"acc": 0.0}
        gold_index = self._get_correct_answer_index(doc, self._get_choices(doc))
        pred_index = np.argmax(results)
        is_correct = (pred_index == gold_index)
        return {"acc": 1.0 if is_correct else 0.0}

# IF YOU WANT TO SEE DEBUG, PLEASE UNCOMMENT fewshot_context(...)
    # def fewshot_context(self, doc: Dict, num_fewshot: int, **kwargs) -> str:
    #     """
    #     Builds the few-shot context and prints it for debugging if num_fewshot > 0.
    #     All debug output is now in English.
    #     """
    #     # This line calls the parent method to actually build the context string
    #     ctx = super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)
    
    #     # We only print the debug output if few-shot examples are being used
    #     if num_fewshot > 0:
    #         print("\n" + "="*50)
    #         # The line below has been changed to English
    #         print(f" DEBUG: Final few-shot context for task '{self.config.task}'")
    #         print("="*50 + f"\n{ctx}\n" + "="*50 + "\n")
        
    #     return ctx


# --- Task class for MMLU-style multiple-choice questions ---
class KyrgyzMMLUTask(KyrgyzBaseTask):
    
    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        # Formats the prompt with question and choices
        question = doc["Суроо (KG)"]
        raw_choices = []
        for choice_key in ["А (KG)", "Б (KG)", "В (KG)", "Г (KG)", "Д (KG)"]:
            if choice_key in doc and doc[choice_key] and str(doc[choice_key]).strip():
                raw_choices.append(str(doc[choice_key]).strip())

        letters = ["А", "Б", "В", "Г", "Д"]
        prompt = f"Суроо: {question}\n\n"
        for i, choice in enumerate(raw_choices):
            prompt += f"{letters[i]}. {choice}\n"
        prompt += "\nЖооп:"
        return prompt

    def doc_to_target(self, doc: Dict[str, Any]) -> str:
        # Gets the correct choice text. Includes a fix for IndexError.
        choices = self._get_choices(doc)
        correct_index = self._get_correct_answer_index(doc, choices)
        if correct_index == -1 or correct_index >= len(choices):
            return ""
        return choices[correct_index]


# --- Task class for Reading Comprehension questions ---
class KyrgyzReadingComprehensionTask(KyrgyzBaseTask):

    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        # Formats the prompt with context, question, and choices
        text = doc["Текст (KG)"]
        question = doc["Суроо (KG)"]
        raw_choices = []
        for choice_key in ["А (KG)", "Б (KG)", "В (KG)", "Г (KG)", "Д (KG)"]:
            if choice_key in doc and doc[choice_key] and str(doc[choice_key]).strip():
                raw_choices.append(str(doc[choice_key]).strip())
        
        letters = ["А", "Б", "В", "Г", "Д"]
        prompt = (
            f"Текстти кунт коюп окуп, суроого туура жооп бериңиз.\n\n"
            f"Текст: {text}\n\n"
            f"Суроо: {question}\n\n"
            "Жооптор:\n"
        )
        for i, choice in enumerate(raw_choices):
            prompt += f"{letters[i]}. {choice}\n"
        prompt += "\nТуура жооп:"
        return prompt

    def doc_to_target(self, doc: Dict[str, Any]) -> str:
        # Gets the correct choice text. Includes a fix for IndexError.
        choices = self._get_choices(doc)
        correct_index = self._get_correct_answer_index(doc, choices)
        if correct_index == -1 or correct_index >= len(choices):
            return ""
        return choices[correct_index]


# --- Register all tasks to be used with the --tasks argument ---
@register_task("kyrgyz_mmlu_all")
class KyrgyzMMLUAll(KyrgyzMMLUTask): DATASET_NAME = "kyrgyz_mmlu_all"
@register_task("kyrgyz_mmlu_history")
class KyrgyzMMLUHistory(KyrgyzMMLUTask): DATASET_NAME = "kyrgyz_mmlu_history"
@register_task("kyrgyz_mmlu_literature")
class KyrgyzMMLULiterature(KyrgyzMMLUTask): DATASET_NAME = "kyrgyz_mmlu_literature"
@register_task("kyrgyz_mmlu_medicine")
class KyrgyzMMLUMedicine(KyrgyzMMLUTask): DATASET_NAME = "kyrgyz_mmlu_medicine"
@register_task("kyrgyz_mmlu_lang")
class KyrgyzMMLULang(KyrgyzMMLUTask): DATASET_NAME = "kyrgyz_mmlu_lang"
@register_task("kyrgyz_mmlu_biology")
class KyrgyzMMLUBiology(KyrgyzMMLUTask): DATASET_NAME = "kyrgyz_mmlu_biology"
@register_task("kyrgyz_mmlu_chemistry")
class KyrgyzMMLUChemistry(KyrgyzMMLUTask): DATASET_NAME = "kyrgyz_mmlu_chemistry"
@register_task("kyrgyz_mmlu_geography")
class KyrgyzMMLUGeography(KyrgyzMMLUTask): DATASET_NAME = "kyrgyz_mmlu_geography"
@register_task("kyrgyz_mmlu_math")
class KyrgyzMMLUMath(KyrgyzMMLUTask): DATASET_NAME = "kyrgyz_mmlu_math"
@register_task("kyrgyz_mmlu_physics")
class KyrgyzMMLUPhysics(KyrgyzMMLUTask): DATASET_NAME = "kyrgyz_mmlu_physics"

@register_task("kyrgyz_rc_all")
class KyrgyzRCAll(KyrgyzReadingComprehensionTask): DATASET_NAME = "kyrgyz_rc_all"
@register_task("kyrgyz_rc_literature")
class KyrgyzRCLiterature(KyrgyzReadingComprehensionTask): DATASET_NAME = "kyrgyz_rc_literature"
@register_task("kyrgyz_rc_math")
class KyrgyzRCMath(KyrgyzReadingComprehensionTask): DATASET_NAME = "kyrgyz_rc_math"
@register_task("kyrgyz_rc_news")
class KyrgyzRCNews(KyrgyzReadingComprehensionTask): DATASET_NAME = "kyrgyz_rc_news"
@register_task("kyrgyz_rc_wiki")
class KyrgyzRCWiki(KyrgyzReadingComprehensionTask): DATASET_NAME = "kyrgyz_rc_wiki"