import os
import io
import gzip
import json
import glob
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Union, Callable
import random
import itertools
Sample = Dict[str, Union[str, float, int, list, dict]]

@dataclass
class FieldMap:
    """Map benchmark-specific fields to the names used by inference."""

    id: List[str] = None
    question: List[str] = None
    answer: List[str] = None
    video_path: List[str] = None
    duration: List[str] = None
    question_type: List[str] = None
    options: List[str] = None
    image: List[str] = None
    subtitle_path: List[str] = None
    extra: Dict[str, List[str]] = None

    def __post_init__(self):
        self.id = self.id or ["id"]
        self.question = self.question or ["question"]
        self.answer = self.answer or ["golden_answer", "answer"]
        self.video_path = self.video_path or ["video_path", "video_path_abs"]
        self.duration = self.duration or ["duration", "video_duration"]
        self.question_type = self.question_type or ["question_type"]
        self.options = self.options or ["options", "choices"]
        self.image = self.image or ["image"]
        self.subtitle_path = self.subtitle_path or ["subtitle_path"]
        self.extra = self.extra or {}

class BaseDatasetReader(ABC):
    """Extensible benchmark reader interface."""

    def __init__(
        self,
        files: Union[str, List[str]],
        field_map: Optional[FieldMap] = None,
        validate: bool = True,
        stream: bool = False,
        postprocess: Optional[Callable[[Sample], Optional[Sample]]] = None,
        video_root: Optional[str] = None,
        subtitle_root: Optional[str] = None,
        *,
        shuffle: bool = False,
        shuffle_seed: Optional[int] = None,
        shuffle_buffer_size: int = 2048,
    ):
        """Create a reader over one or more JSON/JSONL files."""
        self.files = self._expand_files(files)
        if not self.files:
            raise FileNotFoundError(f"No files matched: {files}")
        self.field_map = field_map or FieldMap()
        self.validate = validate
        self.stream = stream
        self.postprocess = postprocess
        self._cache: Optional[List[Sample]] = None
        self.video_root = video_root
        self.subtitle_root = subtitle_root
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.shuffle_buffer_size = max(1, int(shuffle_buffer_size))
        self._rng = random.Random(shuffle_seed)

        if not self.stream:
            self._cache = list(self._iter_impl())
            if self.shuffle and self._cache:
                self._rng.shuffle(self._cache)

    @staticmethod
    def _expand_files(files: Union[str, List[str]]) -> List[str]:
        if isinstance(files, str):
            files = [files]
        paths: List[str] = []
        for f in files:
            matched = glob.glob(f)
            if matched:
                paths.extend(os.path.abspath(p) for p in matched)
            elif os.path.exists(f):
                paths.append(os.path.abspath(f))
        return list(dict.fromkeys(paths))

    def __len__(self) -> int:
        if self.stream:
            raise TypeError("Streaming mode doesn't support len().")
        return len(self._cache or [])

    def __iter__(self) -> Iterator[Sample]:
        if self.stream:
            it = self._iter_impl()
            if self.shuffle:
                yield from self._iter_shuffled_stream(it)
            else:
                yield from it
        else:
            yield from (self._cache or [])

    def _iter_shuffled_stream(self, it: Iterable[Sample]) -> Iterator[Sample]:
        """Shuffle a stream with a bounded buffer."""
        buf_size = self.shuffle_buffer_size
        rng = self._rng

        buffer: List[Sample] = []
        for item in itertools.islice(it, buf_size):
            buffer.append(item)

        for item in it:
            if not buffer:
                buffer.append(item)
            else:
                j = rng.randrange(len(buffer))
                yield buffer.pop(j)
                buffer.append(item)

        while buffer:
            j = rng.randrange(len(buffer))
            yield buffer.pop(j)

    def select_range(self, start: int = 0, end: int = -1) -> "BaseDatasetReader":
        """Return a sliced reader view."""
        if not self.stream:
            data = self._cache or []
            if end is None or end < 0:
                end = len(data)
            sliced = data[start:end]
            clone = self._shallow_clone()
            clone._cache = sliced
            return clone

        parent = self
        class _SliceReader(BaseDatasetReader):
            def __init__(self):
                pass
            def _iter_impl(self) -> Iterator[Sample]:
                for idx, ex in enumerate(parent._iter_impl()):
                    if idx < start:
                        continue
                    if end >= 0 and idx >= end:
                        break
                    yield ex
            def _shallow_clone(self):
                return self
        return _SliceReader()

    def map_fields(self, mapper: Callable[[Sample], Sample]) -> "BaseDatasetReader":
        """Return a mapped reader view."""
        parent = self
        class _MapReader(BaseDatasetReader):
            def __init__(self):
                pass
            def _iter_impl(self) -> Iterator[Sample]:
                for ex in parent._iter_impl():
                    out = mapper(ex)
                    if out is not None:
                        yield out
            def _shallow_clone(self):
                return self
        if not self.stream:
            clone = self._shallow_clone()
            clone._cache = [m for x in (self._cache or []) if (m := mapper(x)) is not None]
            return clone
        return _MapReader()

    def _normalize_fields(self, raw: Sample) -> Optional[Sample]:
        """Normalize a raw benchmark sample to inference field names."""
        def pick(keys: List[str]):
            for k in keys:
                if k in raw and raw[k] not in (None, ""):
                    return raw[k]
            return None

        sample: Sample = dict(raw)

        i = pick(self.field_map.id)
        q = pick(self.field_map.question)
        a = pick(self.field_map.answer)
        vp = pick(self.field_map.video_path)
        sp = pick(self.field_map.subtitle_path)
        du = pick(self.field_map.duration)
        qt = pick(self.field_map.question_type)
        options = pick(self.field_map.options)
        image = pick(self.field_map.image)

        if i is not None:
            sample["id"] = str(i)
        if q is not None:
            sample["question"] = q
        if a is not None:
            sample["golden_answer"] = a
        if vp is not None:
            if isinstance(vp, str) and not vp.lower().endswith(".mp4"):
                vp = vp + ".mp4"
            
            if isinstance(vp, str) and self.video_root and not os.path.isabs(vp):
                vp = os.path.join(self.video_root, vp)
            if isinstance(sp, str) and self.subtitle_root and not os.path.isabs(sp):
                sp = os.path.join(self.subtitle_root, sp)
            sample["video_path"] = vp
        if du is not None:
            try:
                sample["duration"] = float(du)
            except Exception:
                sample["duration"] = du
        if qt is not None:
            sample["question_type"] = qt
        if sp is not None:
            sample["subtitle_path"] = sp

        if options and qt != "open-ended":
            if isinstance(options, list):
                opt_text = ", ".join(str(opt) for opt in options)
            elif isinstance(options, dict):
                opt_text = ", ".join(f"{k}: {v}" for k, v in options.items())
            else:
                opt_text = str(options)
            q_base = sample.get("question", "")
            sample["question"] = f"{q_base}\nOptions: {opt_text}".strip()

        if image:
            image_url = image.get('bytes')
            sample['image_url'] = [image_url]
            
        if self.validate:
            missing = []
            for key in ("question", "video_path"):
                if key not in sample or sample[key] in (None, ""):
                    missing.append(key)
            if missing:
                return None

        if self.postprocess:
            sample = self.postprocess(sample)
        return sample

    @abstractmethod
    def _iter_impl(self) -> Iterator[Sample]:
        ...

    @abstractmethod
    def _shallow_clone(self) -> "BaseDatasetReader":
        ...

class JSONLReader(BaseDatasetReader):
    """Read .jsonl, .jsonl.gz, or .json benchmark files."""

    def _open_file(self, path: str):
        if path.endswith(".gz"):
            return gzip.open(path, "rb")
        return open(path, "rb")

    def _iter_impl(self) -> Iterator[Sample]:
        for path in self.files:
            with self._open_file(path) as fh:
                text_fh = io.TextIOWrapper(fh, encoding="utf-8")
                if path.endswith(".jsonl") or path.endswith(".gz"):
                    for ln in text_fh:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            raw = json.loads(ln)
                            norm = self._normalize_fields(raw)
                            if norm is not None:
                                yield norm
                        except json.JSONDecodeError:
                            continue
                else:
                    try:
                        raw_data = json.load(text_fh)
                        if isinstance(raw_data, list):
                            for item in raw_data:
                                norm = self._normalize_fields(item)
                                if norm is not None:
                                    yield norm
                        else:
                            norm = self._normalize_fields(raw_data)
                            if norm is not None:
                                yield norm
                    except json.JSONDecodeError:
                        continue


    def _shallow_clone(self) -> "JSONLReader":
        clone = JSONLReader(
            files=self.files,
            field_map=self.field_map,
            validate=self.validate,
            stream=self.stream,
            postprocess=self.postprocess,
            video_root=self.video_root,
            subtitle_root=self.subtitle_root,
        )
        clone._cache = self._cache
        return clone
