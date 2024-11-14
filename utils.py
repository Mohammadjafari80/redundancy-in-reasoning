# utils.py

import sys
from io import StringIO
import logging
import torch
import math
import datetime

def tokenize_function(examples, tokenizer, max_length=512):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=max_length,
    )

import math
import datetime
import sys
from io import StringIO
import signal
import logging
import ast
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Code execution timed out")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)

class LastAssignmentFinder(ast.NodeVisitor):
    def __init__(self):
        self.last_assigned_var = None
        self.has_return = False
    
    def visit_Assign(self, node):
        # Track only simple variable assignments
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self.last_assigned_var = node.targets[0].id
        self.generic_visit(node)
    
    def visit_Return(self, node):
        self.has_return = True
        self.generic_visit(node)

def filter_code_to_function_only(code, func_name="solution"):
    """
    Filters the code to include only the function body by using indentation levels.
    """
    lines = code.splitlines()
    function_found = True
    filtered_lines = []
    function_indent = None
    function_indent = len(lines[0]) - len(lines[0].lstrip())
    
    for line in lines:
        stripped_line = line.strip()

        if function_found:
            current_indent = len(line) - len(line.lstrip())
            if function_indent is not None and current_indent < function_indent and stripped_line:
                # Stop processing lines when encountering less-indented code after function starts
                break
            filtered_lines.append(line)
    
    return "\n".join(filtered_lines)

def analyze_solution_function(code):
    try:
        # Parse the code into an AST
        tree = ast.parse(code)
        
        # Find the solution function definition
        solution_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'solution':
                solution_func = node
                break
        
        if solution_func is None:
            return None, False
        
        # Analyze the function body
        finder = LastAssignmentFinder()
        finder.visit(solution_func)
        
        return finder.last_assigned_var, finder.has_return
    except:
        return None, False

def extract_final_answer_from_code(code):
    """
    Executes the generated code and captures either the return value or the last assigned variable.
    """
    try:
        old_stdout = sys.stdout
        
        # Filter the code to keep only the solution function
        code = filter_code_to_function_only(code)
        
        # Remove all print statements from the code
        code_lines = code.splitlines()
        code_lines_no_print = [line for line in code_lines if 'print(' not in line and 'input(' not in line]
        code = '\n'.join(code_lines_no_print)
        
        # Prepare the code for execution
        import_statements = "import math\nimport datetime\n"
        def_function = "def solution():\n"
        # Indent the user code to fit inside the function
        indented_code = '\n'.join(line for line in code.splitlines() if "input(" not in line)
        
        # Analyze the function
        last_var, has_return = analyze_solution_function(def_function + indented_code)
        
        if has_return:
            # Use original execution approach
            code_to_exec = import_statements + def_function + indented_code + "\nresult = solution()"
        elif last_var:
            # Modify function to return the last assigned variable
            code_to_exec = (
                import_statements + 
                def_function + 
                indented_code + 
                f"\n    return {last_var}\n" +
                "result = solution()"
            )
        else:
            # No return and no assignments found
            return None

        # Redirect stdout to capture print output
        sys.stdout = mystdout = StringIO()
        
        # Use default built-ins and include necessary modules
        exec_globals = {
            '__builtins__': __builtins__,
            'math': math,
            'datetime': datetime,
        }
        
        # Execute the code with timeout
        with timeout(1):
            exec(code_to_exec, exec_globals)
            
        # Get the result
        result = exec_globals.get('result', None)
        
        # Return the result, ignoring any printed output since we're focusing on returns/assignments
        if result is not None:
            return str(result)
        return None 
    except TimeoutException:
        logging.warning("Code execution timed out after 1 second")
        return None
    except Exception as e:
        logging.warning(f"Execution error: {e}")
        return None
    finally:
        # Restore stdout
        sys.stdout = old_stdout

def generate_answer(
    examples, tokenizer, model, device, num_beams=3, temperature=0.7, max_length=512
):
    input_texts = examples['text']
    
    # Tokenize inputs
    inputs = tokenizer(
        input_texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length, 
    ).to(device)
    
    assert inputs['input_ids'].shape == inputs['attention_mask'].shape, "Input IDs and attention mask must have the same shape"
    
    with torch.no_grad():
        if device.type == 'cuda':
            # with torch.amp.autocast('cuda'):
            output_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'], 
                max_length=max_length, 
                num_beams=num_beams,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            output_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'], 
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.eos_token_id,
            )
    
    generated_codes = []
    for idx in range(len(input_texts)):
        generated_code = tokenizer.decode(output_ids[idx], skip_special_tokens=True)
        generated_code = generated_code[len(input_texts[idx]):]
        filtered_generated_code = ''
        for generated_line in generated_code.split('\n'):
            filtered_generated_code += f'{generated_line}\n'
            if 'return' in generated_line:
                break
        generated_codes.append(filtered_generated_code)

    examples['generated_code'] = generated_codes
    return examples

def evaluate(example):
    generated_code = example['generated_code']
    final_answer = extract_final_answer_from_code(generated_code)
    example['predicted_answer'] = final_answer
    return example