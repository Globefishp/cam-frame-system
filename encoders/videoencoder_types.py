class EncoderException(Exception):
    def __init__(self, message: str, pid: int, name: str):
        self.message = message
        self.pid = pid
        self.name = name
        super().__init__(f"[{pid:>5}] {name}: {message}")