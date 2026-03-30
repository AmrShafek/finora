from .base_validator import Validator


class PatternValidator(Validator):
    def validate(self, output: str) -> dict:
        if output and len(output.strip()) > 10:
            return {"valid": True, "issues": []}
        return {"valid": False, "issues": ["Pattern output too short or empty"]}
