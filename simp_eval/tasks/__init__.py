from .drop_eval import DropEval
from .gpqa_eval import GPQAEval
from .humaneval_eval import HumanEval
from .math_eval import MathEval
from .mgsm_eval import MGSMEval
from .mmlu_eval import MMLUEval

TASK_REGISTRY = {
    "mmlu": MMLUEval,
    "math": MathEval,
    "gpqa": GPQAEval,
    "mgsm": MGSMEval,
    "drop": DropEval,
    "humaneval": HumanEval,
}
ALL_TASKS = sorted(list(TASK_REGISTRY))
