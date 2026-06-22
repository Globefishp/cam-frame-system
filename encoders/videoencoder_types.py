from typing import Protocol, Any, List, Dict
from numpy.typing import NDArray

class EncoderException(Exception):
    def __init__(self, message: str, pid: int, name: str):
        self.message = message
        self.pid = pid
        self.name = name
        super().__init__(f"[{pid:>5}] {name}: {message}")

class TimecodeExtractor(Protocol):
    def __call__(self, image: NDArray, **kwargs: Any) -> tuple[NDArray, List[Dict[str, Any]]]:
        pass
    @property
    def timebase(self) -> int:
        """1/timebase equals time per tick in second."""
        pass
    @property
    def timecode_key(self) -> str:
        """The key to extract timecode from the extended info dict."""
        pass
    # TODO: Wrapper for ExtInfoExtractor, done in Backend (top level that responsible for DI)