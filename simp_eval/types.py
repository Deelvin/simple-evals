from dataclasses import dataclass, field
from typing import Any
from omegaconf import OmegaConf

Message = dict[str, Any]  # keys role, content
MessageList = list[Message]


class SamplerBase:
    """
    Base class for defining a sampling model, which can be evaluated,
    or used as part of the grading process.
    """

    def __call__(self, message_list: MessageList) -> str:
        raise NotImplementedError

    @classmethod
    def create_from_arg_string(cls, args_string):
        args_string = args_string.strip()
        if not args_string:
            return {}
        arg_list = args_string.split(",")
        args_dict = OmegaConf.to_object(OmegaConf.from_dotlist(arg_list))
        return cls(**args_dict)


@dataclass
class EvalResult:
    """
    Result of running an evaluation (usually consisting of many samples)
    """

    score: float | None  # top-line metric
    metrics: dict[str, float] | None  # other metrics
    htmls: list[str]  # strings of valid HTML
    convos: list[MessageList]  # sampled conversations


@dataclass
class SingleEvalResult:
    """
    Result of evaluating a single sample
    """

    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    html: str | None = None
    convo: MessageList | None = None  # sampled conversation


class Eval:
    """
    Base class for defining an evaluation.
    """

    def __call__(self, sampler: SamplerBase, batch_size: int) -> EvalResult:
        raise NotImplementedError
