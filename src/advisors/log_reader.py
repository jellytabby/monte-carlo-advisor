"""Reader for training log.

See lib/Analysis/TrainingLogger.cpp for a description of the format.
"""

import ctypes
import dataclasses
import io
import json
import math
import sys
import numpy as np
from typing import List, Optional, Union

_element_types = {
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    "int8_t": ctypes.c_int8,
    "uint8_t": ctypes.c_uint8,
    "int16_t": ctypes.c_int16,
    "uint16_t": ctypes.c_uint16,
    "int32_t": ctypes.c_int32,
    "uint32_t": ctypes.c_uint32,
    "int64_t": ctypes.c_int64,
    "uint64_t": ctypes.c_uint64,
}


def send(f: io.BufferedWriter, value: Union[int, float], spec: Optional["TensorSpec"]):
    """Send the `value` - currently just a scalar - formatted as per `spec`."""
    assert spec

    if spec.element_type == ctypes.c_int64:
        convert_el_func = int
        ctype_func = ctypes.c_int64
    elif spec.element_type == ctypes.c_float:
        convert_el_func = float
        ctype_func = ctypes.c_float
    else:
        print(spec.element_type, "not supported")
        assert False

    if isinstance(value, list):
        to_send = (ctype_func * len(value))(*[convert_el_func(el) for el in value])
    else:
        to_send = ctype_func(convert_el_func(value))

    assert f.write(bytes(to_send)) == ctypes.sizeof(spec.element_type) * math.prod(
        spec.shape
    )
    f.flush()


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    name: str
    port: int
    shape: List[int]
    element_type: type

    @staticmethod
    def from_dict(d: dict):
        name = d["name"]
        port = d["port"]
        shape = [int(e) for e in d["shape"]]
        element_type_str = d["type"]
        if element_type_str not in _element_types:
            raise ValueError(f"uknown type: {element_type_str}")
        return TensorSpec(
            name=name,
            port=port,
            shape=shape,
            element_type=_element_types[element_type_str],
        )


class TensorValue:
    def __init__(self, spec: TensorSpec, buffer: bytes):
        self._spec = spec
        self._buffer = buffer
        self._view = ctypes.cast(self._buffer, ctypes.POINTER(self._spec.element_type))
        self._len = math.prod(self._spec.shape)

    def spec(self) -> TensorSpec:
        return self._spec

    def to_numpy(self) -> np.ndarray:
        # TODO our unroll model currently expects only signed int inputs -
        # interpret unsigned ints as signed ones for now
        el_ty = self._spec.element_type
        if self._spec.element_type == ctypes.c_ulong:
            el_ty = ctypes.c_long
        return np.frombuffer(
            self._buffer, dtype=el_ty, count=self._len)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index):
        if index < 0 or index >= self._len:
            raise IndexError(f"Index {index} out of range [0..{self._len})")
        return self._view[index]


def read_tensor(fs: io.BufferedReader, ts: TensorSpec) -> TensorValue:
    size = math.prod(ts.shape) * ctypes.sizeof(ts.element_type)
    data = fs.read(size)
    return TensorValue(ts, data)


def string_tensor_value(tv: TensorValue) -> str:
    return f'{tv.spec().name}: {",".join([str(v) for v in tv])}'


def pretty_print_tensor_value(tv: TensorValue):
    print(string_tensor_value(tv))


def read_header(f: io.BufferedReader):
    header = json.loads(f.readline())
    tensor_specs = [TensorSpec.from_dict(ts) for ts in header["features"]]
    score_spec = TensorSpec.from_dict(header["score"]) if "score" in header else None
    advice_spec = TensorSpec.from_dict(header["advice"]) if "advice" in header else None
    return header, tensor_specs, score_spec, advice_spec


def read_one_observation(
    context: Optional[str],
    event_str: bytes,
    f: io.BufferedReader,
    tensor_specs: List[TensorSpec],
    score_spec: Optional[TensorSpec],
):
    event = json.loads(event_str)
    if "context" in event:
        context = event["context"]
        event = json.loads(f.readline())
    observation_id = int(event["observation"])
    features = []
    for ts in tensor_specs:
        features.append(read_tensor(f, ts))
    f.readline()
    score = None
    if score_spec is not None:
        score_header = json.loads(f.readline())
        assert int(score_header["outcome"]) == observation_id
        score = read_tensor(f, score_spec)
        f.readline()
    return context, observation_id, features, score


def read_stream(fname: str):
    with io.BufferedReader(io.FileIO(fname, "rb")) as f:
        _, tensor_specs, score_spec, _ = read_header(f)
        context = None
        while True:
            event_str = f.readline()
            if not event_str:
                break
            context, observation_id, features, score = read_one_observation(
                context, event_str, f, tensor_specs, score_spec
            )
            yield context, observation_id, features, score


def main(args):
    last_context = None
    for ctx, obs_id, features, score in read_stream(args[1]):
        if last_context != ctx:
            print(f"context: {ctx}")
            last_context = ctx
        print(f"observation: {obs_id}")
        for fv in features:
            pretty_print_tensor_value(fv)
        if score:
            pretty_print_tensor_value(score)


if __name__ == "__main__":
    main(sys.argv)
