import json
from .base_validator import Validator


class OCRValidator(Validator):
    def validate(self, output: str) -> dict:
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return {"valid": False, "issues": ["Output is not valid JSON"]}

        required_fields = ["date", "amount", "category", "description"]
        issues = []

        for field in required_fields:
            if field not in data:
                issues.append(f"Missing required field: {field}")

        if "amount" in data:
            try:
                float(data["amount"])
            except (ValueError, TypeError):
                issues.append("Amount must be a valid number")

        if "category" in data:
            valid_categories = ["Salaries", "Software", "Marketing", "Operations", "Travel", "Utilities", "Other"]
            if data["category"] not in valid_categories:
                issues.append(f"Invalid category: {data['category']}. Must be one of: {valid_categories}")

        if issues:
            return {"valid": False, "issues": issues}

        return {"valid": True, "issues": []}
