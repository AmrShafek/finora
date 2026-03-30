from .base_validator import Validator


class ForecastValidator(Validator):
    def validate(self, output: str) -> dict:
        # Simple heuristic: require presence of 'forecast' keyword or numeric trends
        ok = ("forecast" in output.lower()) or any(ch.isdigit() for ch in output)
        if ok:
            return {"valid": True, "issues": []}
        return {"valid": False, "issues": ["Forecast output seems empty or not in expected format"]}
