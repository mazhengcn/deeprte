"""Splits related API."""

import dataclasses
import math
import re

# <split_name>[<split_selector>] (e.g. `train[54%:]`)
_SUB_SPEC_RE = re.compile(
    r"""^
    (?P<split_name>[\w-]+)
    (\[
      (?P<split_selector>[\d\w%:.-]+)
    \])?
    $""",
    re.X,  # Ignore whitespace
)
# <val><unit> (e.g. `-54%`)
_SLICE_RE = re.compile(
    r"""^
    (
        (?P<val>-?[\d_.]+)
        (?P<unit>(?:%|shard))?
    )?
    $""",
    re.X,  # Ignore whitespace
)

_ADDITION_SEP_RE = re.compile(r"\s*\+\s*")


@dataclasses.dataclass(frozen=True)
class AbsoluteInstruction:
    """A machine friendly slice: defined absolute positive boundaries."""

    splitname: str
    from_: int  # uint (starting index).
    to: int  # uint (ending index).


@dataclasses.dataclass(frozen=True)
class ReadInstruction:
    split_name: str
    # TODO(py3.10): Add `_ = dataclasses.KW_ONLY`
    from_: int | float | None = None
    to: int | float | None = None
    unit: str = "abs"
    rounding: str = "closest"

    def __post_init__(self):
        # Perform validation
        allowed_units = ["%", "abs", "shard"]
        allowed_rounding = ["closest", "pct1_dropremainder"]
        if self.unit not in allowed_units:
            raise ValueError(
                f"Unit should be one of {allowed_units}. Got {self.unit!r}"
            )
        if self.rounding not in allowed_rounding:
            raise ValueError(
                f"Rounding should be one of {allowed_rounding}. Got: {self.rounding!r}"
            )
        if self.unit == "%":
            if abs(self.from_ or 0) > 100 or abs(self.to or 0) > 100:
                raise ValueError(
                    "When unit=%, percent slice boundaries should be "
                    f"in [-100, 100]. Got: {self}"
                )

    def __repr__(self) -> str:
        unit = "" if self.unit == "abs" else self.unit
        from_ = "" if self.from_ is None else f"{self.from_:g}{unit}"
        to = "" if self.to is None else f"{self.to:g}{unit}"
        if self.from_ is None and self.to is None:
            slice_str = ""  # Full split selected
        else:
            slice_str = f"[{from_}:{to}]"
        rounding = f", rounding={self.rounding!r}" if self.unit == "%" else ""
        return f"ReadInstruction('{self.split_name}{slice_str}'{rounding})"


def get_split_instruction(spec: str, num_examples: int) -> AbsoluteInstruction:
    """Parse a split string into an AbsoluteInstruction."""
    rel_instr = _str_to_relative_instruction(spec)
    return _rel_to_abs_instr(rel_instr, num_examples)


def _str_to_relative_instruction(spec: str):
    """Returns ReadInstruction for given string."""
    # <split_name>[<split_selector>] (e.g. `train[54%:]`)
    res = _SUB_SPEC_RE.match(spec)
    err_msg = (
        f"Unrecognized split format: {spec!r}. See format at "
        "https://www.tensorflow.org/datasets/splits"
    )
    if not res:
        raise ValueError(err_msg)
    split_name = res.group("split_name")
    split_selector = res.group("split_selector")

    if split_name == "all":
        if split_selector:
            # TODO(tfds): `all[:75%]` could be supported by creating a
            # `_SliceSplit(split, from_=, to=, unit=)`.
            raise NotImplementedError(
                f"{split_name!r} does not support slice. Please open a github issue "
                "if you need this feature."
            )
        split_selector = None

    if split_selector is None:  # split='train'
        from_ = None
        to = None
        unit = "abs"
    else:  # split='train[x:y]' or split='train[x]'
        slices = [_SLICE_RE.match(x) for x in split_selector.split(":")]
        # Make sure all slices are valid, and at least one is not empty
        if not all(slices) or not any(
            x.group(0) for x in slices if x is not None
        ):  # re-none
            raise ValueError(err_msg)
        if len(slices) == 1:  # split='train[x]'
            (from_match,) = slices
            from_ = from_match["val"]
            to = int(from_) + 1
            unit = from_match["unit"] or "abs"
            if unit != "shard":
                raise ValueError("Absolute or percent only support slice syntax.")
        elif len(slices) == 2:  # split='train[x:y]'
            from_match, to_match = slices
            from_ = from_match["val"]
            to = to_match["val"]
            unit = from_match["unit"] or to_match["unit"] or "abs"
        else:
            raise ValueError(err_msg)

    if from_ is not None:
        from_ = float(from_) if unit == "%" else int(from_)
    if to is not None:
        to = float(to) if unit == "%" else int(to)

    return ReadInstruction(
        split_name=split_name,
        rounding="closest",
        from_=from_,
        to=to,
        unit=unit,
    )


def _pct_to_abs_pct1(boundary, num_examples: int):
    # Using math.trunc here, since -99.5% should give -99%, not -100%.
    if num_examples < 100:
        msg = (
            'Using "pct1_dropremainder" rounding on a split with less than 100 '
            "elements is forbidden: it always results in an empty dataset."
        )
        raise ValueError(msg)
    return boundary * math.trunc(num_examples / 100.0)


def _pct_to_abs_closest(boundary, num_examples: int) -> int:
    return int(round(boundary * num_examples / 100.0))


def _rel_to_abs_instr(
    rel_instr: ReadInstruction, num_examples: int
) -> AbsoluteInstruction:
    """Returns _AbsoluteInstruction instance for given RelativeInstruction.

    Args:
      rel_instr: ReadInstruction instance.
      split_infos: dict {split_name: split_infos}.
    """
    pct_to_abs = (
        _pct_to_abs_closest if rel_instr.rounding == "closest" else _pct_to_abs_pct1
    )
    split = rel_instr.split_name
    from_ = rel_instr.from_
    to = rel_instr.to
    if rel_instr.unit == "%":
        from_ = 0 if from_ is None else pct_to_abs(from_, num_examples)
        to = num_examples if to is None else pct_to_abs(to, num_examples)
    elif rel_instr.unit == "abs":
        from_ = 0 if from_ is None else from_
        to = num_examples if to is None else to
    else:
        raise ValueError(f"Invalid split unit: {rel_instr.unit}")
    if abs(from_) > num_examples or abs(to) > num_examples:
        msg = "Requested slice [%s:%s] incompatible with %s examples." % (
            from_ or "",
            to or "",
            num_examples,
        )
        raise ValueError(msg)
    if from_ < 0:
        from_ = num_examples + from_
    elif from_ == 0:
        from_ = None
    if to < 0:
        to = num_examples + to
    elif to == num_examples:
        to = None
    return AbsoluteInstruction(split, from_, to)
