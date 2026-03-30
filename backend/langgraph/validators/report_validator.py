from .base_validator import Validator


class ReportValidator(Validator):
    def validate(self, output: str) -> dict:
        if output and len(output.strip()) > 20:
            return {"valid": True, "issues": []}
        return {"valid": False, "issues": ["Report output too short or empty"]}
