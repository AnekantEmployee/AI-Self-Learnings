import re

# Dictionary to map Sapera syntax to Python (updated)
sapera_keywords = {
    r'ye le (\w+) = (.+)': r'\1 = \2',  # Variable declaration
    r'bolo (.+)': r'print(\1)',         # Print statement
    r'agar (.+) toh': r'if \1:',        # If condition
    r'warna toh': r'else:',             # Else condition
    r'bas kar bhai': r'break',          # Break statement
    r'ghoom ke aao jab tak (.+)': r'while \1:',  # Fixed: No extra colon
}

def sapera_to_python(code):
    for pattern, replacement in sapera_keywords.items():
        code = re.sub(pattern, replacement, code)
    return code

# Example Sapera Code
sapera_code = """
ye le x = 10
agar x == 10 toh
    bolo "Sahi hai!"
warna toh
    bolo "Galat hai!"
ghoom ke aao jab tak x > 0
    bolo x
    ye le x = x - 1
    agar x == 5 toh
        bas kar bhai
"""

python_code = sapera_to_python(sapera_code)
print("Generated Python Code:\n", python_code)

# Execute the translated Python code
exec(python_code)