import argparse
from datetime import timedelta

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.utils.imports import is_accelerate_available

if is_accelerate_available():
    from datetime import timedelta
    from accelerate import Accelerator, InitProcessGroupKwargs
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLMs with LightEval")

    parser.add_argument("--model", required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--local_model", action="store_true", help="Use local model path instead of HF")
    parser.add_argument("--bench", required=True, choices=["mmlu", "lswag", "winogrande", "gsm", "truthfulqa", "boolq"], help="Benchmark to run")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--mmlu_subject", type=str, help="Specific MMLU subject to test. If not provided, all subjects will be tested")

    args = parser.parse_args()

    # eval tracker
    evaluation_tracker = EvaluationTracker(
        output_dir="lighteval/results_english",
        save_details=True,
        push_to_hub=False
    )

    #  pipeline parameters
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        # env_config=EnvConfig(cache_dir="./tmp"),
        max_samples=None
    )

    # Model config
    model_config = TransformersModelConfig(
        model_name=args.model,
        device="cuda",  # or "auto"
        use_chat_template=True,
        batch_size=args.batch_size
    )

    if args.local_model:
        model_config.load_from_local = True

    
    # рandle MMLU otdelno ot other benchmarks
    if args.bench == "mmlu":
        if args.mmlu_subject:
            # If specific subject provided, test only that subject
            task_id = f"helm|mmlu:{args.mmlu_subject}|{args.num_fewshot}|0"
        else:
            # If no subject provided, тестит все сабджекты по дефолту
            task_id = f"helm|mmlu|{args.num_fewshot}|0"
    else:
        # leaderboard for non-MMLU tasks
        task_map = {
            "winogrande": f"leaderboard|winogrande|{args.num_fewshot}|0",
            "lswag": f"leaderboard|hellaswag|{args.num_fewshot}|0",
            "gsm": f"leaderboard|gsm8k|{args.num_fewshot}|0",
            "truthfulqa": f"leaderboard|truthfulqa|{args.num_fewshot}|0",
            "boolq": f"leaderboard|boolq|{args.num_fewshot}|0",
        }
        task_id = task_map[args.bench]

    print(f"\n=== Running {args.bench.upper()} on {args.model} ===")
    if args.bench == "mmlu":
        print(f"MMLU Subject: {args.mmlu_subject if args.mmlu_subject else 'all subjects'}")
        
    pipeline = Pipeline(
        tasks=task_id,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()

if __name__ == "__main__":
    main()