import re
from .base_validator import Validator


class DataValidator(Validator):
    def validate(self, output: str) -> dict:
        # Expect sections 1., 2., 3. as per current data_agent prompt
        has_1 = bool(re.search(r"^\s*1\.", output, re.MULTILINE))
        has_2 = bool(re.search(r"^\s*2\.", output, re.MULTILINE))
        has_3 = bool(re.search(r"^\s*3\.", output, re.MULTILINE))
        if has_1 and has_2 and has_3:
            return {"valid": True, "issues": []}
        issues = []
        if not has_1:
            issues.append("Missing section 1: data specification")
        if not has_2:
            issues.append("Missing section 2: time period")
        if not has_3:
            issues.append("Missing section 3: metrics focus")
        return {"valid": False, "issues": issues}
