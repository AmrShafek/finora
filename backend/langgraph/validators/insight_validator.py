from .base_validator import Validator


class InsightValidator(Validator):
    def validate(self, output: str) -> dict:
        # Minimal: ensure some sentiment/structure present
        if output and len(output.strip()) > 20:
            return {"valid": True, "issues": []}
        return {"valid": False, "issues": ["Insight output too short or empty"]}
