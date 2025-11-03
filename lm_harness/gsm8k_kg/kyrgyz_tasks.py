import numpy as np
import re
from typing import List, Dict, Any
from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_task
from lm_eval.api.filter import Filter


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


# --- Custom Filters for GSM8K ---
class FlexibleExtractFilter(Filter):
    """Extracts numerical answers flexibly from generated text."""
    
    def __init__(self) -> None:
        pass
    
    def apply(self, resps, docs):
        def extract_answer(text: str) -> str:
            """Extracts the final numerical answer from a string."""
            # Try to find answer after ####
            match = re.search(r"####\s*(-?[\d,]+)", text)
            if match:
                num_str = match.group(1).replace(",", "")
                return num_str.strip()
            
            # Fallback: get last line and extract numbers
            cleaned_text = text.strip().split("\n")[-1]
            cleaned_text = re.sub(r"[^0-9-]", "", cleaned_text)
            return cleaned_text
        
        filtered_resps = []
        for resp in resps:
            filtered_resps.append(extract_answer(resp))
        return filtered_resps


class StrictMatchFilter(Filter):
    """Returns the exact generated text without modification."""
    
    def __init__(self) -> None:
        pass
    
    def apply(self, resps, docs):
        # Return responses as-is for strict matching
        return [resp.strip() for resp in resps]


# --- Task class for GSM8K-style math questions ---
class KyrgyzGSM8KTask(ConfigurableTask):
    """
    A task for evaluating the Kyrgyz GSM8K benchmark with both flexible and strict matching.
    """
    VERSION = 9  # Updated version
    DATASET_PATH = "TTimur/gsm8k_kg"
    OUTPUT_TYPE = "generate_until"

    def __init__(self, config):
        clean_config = config.copy()
        clean_config.pop("python_file", None)
        clean_config.pop("class", None)
        if "repeats" not in clean_config or clean_config["repeats"] is None:
            clean_config["repeats"] = 1
        super().__init__(config=clean_config)
        self.repeats = clean_config["repeats"]

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        from datasets import load_dataset
        self.dataset = load_dataset(
            path=self.DATASET_PATH,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )

    def has_training_docs(self) -> bool: return True
    def has_validation_docs(self) -> bool: return False
    def has_test_docs(self) -> bool: return True

    def training_docs(self): return self.dataset["train"]
    def validation_docs(self): return []
    def test_docs(self): return self.dataset["test"]

    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        return f"Суроо: {doc['question_kg']}\nЖооп:"

    def doc_to_target(self, doc: Dict[str, Any]) -> str:
        return f" {doc['answer_kg']}"

    def construct_requests(self, doc: Dict[str, Any], ctx: str, **kwargs):
        """
        Creates multiple requests with different filters for flexible and strict matching.
        """
        args = (ctx, {"until": ["\n", "####"]})
        
        # Get metadata if available
        metadata = kwargs.get("metadata", {})
        
        # Create flexible-extract request
        flexible_request = Instance(
            request_type="generate_until",
            doc=doc,
            arguments=args,
            idx=0,
            metadata=metadata,
            repeats=1,
        )
        flexible_request.task_name = "kyrgyz_gsm8k"
        flexible_request.filter_key = "flexible-extract"
        
        # Create strict-match request
        strict_request = Instance(
            request_type="generate_until",
            doc=doc,
            arguments=args,
            idx=0,
            metadata=metadata,
            repeats=1,
        )
        strict_request.task_name = "kyrgyz_gsm8k"
        strict_request.filter_key = "strict-match"
        
        return [flexible_request, strict_request]

    def _extract_answer(self, text: str) -> str:
        """Extracts the final numerical answer from a string."""
        match = re.search(r"####\s*(-?[\d,]+)", text)
        if match:
            num_str = match.group(1).replace(",", "")
            return num_str.strip()
        
        cleaned_text = text.strip().split("\n")[-1]
        cleaned_text = re.sub(r"[^0-9-]", "", cleaned_text)
        return cleaned_text

    def process_results(self, doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
        """
        Processes results for both flexible and strict matching.
        """
        if not results:
            return {"exact_match": 0.0}
            
        prediction = results[0]
        gold_target = doc["answer_kg"]
        
        # Extract the numerical answer from gold target
        gold = self._extract_answer(gold_target)
        
        # For flexible matching, extract from prediction
        pred_flexible = self._extract_answer(prediction)
        
        # For strict matching, use prediction as-is
        pred_strict = prediction.strip()
        gold_strict = gold_target.strip()
        
        is_correct_flexible = (gold == pred_flexible)
        is_correct_strict = (gold_strict == pred_strict)
        
        return {
            "exact_match": float(is_correct_flexible),
            "exact_match_strict": float(is_correct_strict)
        }

    def higher_is_better(self) -> Dict[str, bool]:
        return {
            "exact_match": True,
            "exact_match_strict": True
        }

    def aggregation(self) -> Dict[str, Any]:
        from lm_eval.api.metrics import mean
        return {
            "exact_match": mean,
            "exact_match_strict": mean
        }


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

# New registration for Kyrgyz GSM8K
@register_task("kyrgyz_gsm8k")
class KyrgyzGSM8K(KyrgyzGSM8KTask): pass