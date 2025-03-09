from typing import Tuple
""" This class is used for gaurd rail the input and output . 
If any inappropriate input is sent / hallucinated output is retrieved , it response with default response."""

class Guardrails:
    def __init__(self):
        self.financial_keywords = {
            'revenue', 'profit', 'loss', 'earnings', 'assets', 'liabilities',
            'balance', 'cash flow', 'income', 'expenses', 'debt', 'equity','sales','sold','market','capitalisation','dividend'
        }
        
    def validate_input(self, query: str) -> Tuple[bool, str]:
        """Input-side guardrail."""
        query = query.lower()
        if not any(keyword in query for keyword in self.financial_keywords):
            return False, "Query must be related to financial information."
        if len(query.split()) > 50:
            return False, "Query too long. Please be more concise."
        return True, ""
        
    def validate_output(self, response: str, confidence: float) -> Tuple[bool, str]:
        """Output-side guardrail."""
        if confidence < 0.3:
            return False, "Low confidence in response. Please rephrase the question."
        if len(response) > 500:
            return False, "Response too long. Summarizing key points..."
        return True, ""
