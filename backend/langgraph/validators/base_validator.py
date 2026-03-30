from abc import ABC, abstractmethod


class Validator(ABC):
    @abstractmethod
    def validate(self, output: str) -> dict:
        """Return a dict with at least 'valid': bool and optional 'issues': list[str]"""
        raise NotImplementedError
