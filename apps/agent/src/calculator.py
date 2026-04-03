from langchain_core.tools import tool

@tool
def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.
    Examples: '2 + 2', '10 * 5', '100 / 4', '2 ** 8'
    """
    try:
        # Only allow safe math operations
        allowed = set("0123456789+-*/().**% ")
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
