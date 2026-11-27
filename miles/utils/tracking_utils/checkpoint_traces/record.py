from dataclasses import asdict, dataclass
from typing import Any

from typing_extensions import Self

from miles.utils.types import Sample


@dataclass(frozen=True)
class TraceRecord:
    """One human-readable eval trace: what was asked, what the model answered, and how it scored."""

    step: int
    dataset: str
    prompt_id: int | None
    prompt: str | list[dict[str, str]]
    response: str
    expected: str | None
    reward: float | dict[str, Any] | None
    response_length: int
    truncated: bool
    status: str

    @classmethod
    def from_sample(cls, sample: Sample, *, step: int, dataset: str) -> Self:
        """Keep the readable fields of ``sample``; token ids, log-probs and other arrays are left out."""
        return cls(
            step=step,
            dataset=dataset,
            prompt_id=sample.index,
            prompt=sample.prompt,
            response=sample.response,
            expected=sample.label,
            reward=sample.reward,
            response_length=sample.response_length,
            truncated=sample.status == Sample.Status.TRUNCATED,
            status=sample.status.value,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-ready dict."""
        return asdict(self)
