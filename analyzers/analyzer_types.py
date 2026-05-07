from enum import Enum

class TensorType(Enum):
    NUMPY = 'numpy'
    TORCH = 'torch'

class DeviceType(Enum):
    CPU = 'cpu'
    CUDA = 'cuda'

class ConsumerMode(Enum):
    SYNC = 'sync'
    ASYNC = 'async'

class AnalyzerException(Exception):
    def __init__(self, message: str, pid: int, name: str):
        self.message = message
        self.pid = pid
        self.name = name
        super().__init__(f"[{pid:>5}] {name}: {message}")
