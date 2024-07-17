import json
import argparse
import pandas as pd

from simp_eval import common, tasks, samplers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples per task.",
    )
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    parser.add_argument("--judge_model", default=None)
    parser.add_argument("--batch_size", default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_name = {}
        for task_name in args.tasks.split(","):
            assert (
                task_name in tasks.ALL_TASKS
            ), f"Incorrect name of a task: {task_name}"
        task_names = {
            task_name: tasks.TASK_REGISTRY[task_name](num_examples=args.limit)
        }

    sampler = samplers.get_sampler(args.sampler).create_from_arg_string(args.model_args)
    if "math" in task_names:
        assert (
            args.judge_model is not None
        ), "Provide model for equality check in math tasks"
        equality_checker = sampler(model=args.judge_model)
        task_names["math"].equality_checker = equality_checker

    debug_suffix = "_DEBUG" if args.limit else ""
    mergekey2resultpath = {}
    for eval_name, eval_obj in task_names.items():
        result = eval_obj(sampler, args.batch_size)
        report_filename = f".logs/{eval_name}{debug_suffix}.html"
        print(f"Writing report to {report_filename}")
        with open(report_filename, "w") as fh:
            fh.write(common.make_report(result))
        metrics = result.metrics | {"score": result.score}
        print(metrics)
        result_filename = f".logs/{eval_name}{debug_suffix}.json"
        with open(result_filename, "w") as f:
            f.write(json.dumps(metrics, indent=2))
        print(f"Writing results to {result_filename}")
        mergekey2resultpath[f"{eval_name}"] = result_filename
    merge_metrics = []
    for eval_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        sampler_name = args.sampler
        merge_metrics.append(
            {"eval_name": eval_name, "sampler_name": sampler_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["sampler_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()
