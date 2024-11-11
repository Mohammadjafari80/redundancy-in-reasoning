# utils.py

import sys
from io import StringIO
import logging
import torch

def tokenize_function(examples, tokenizer, max_length=512):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=max_length,
    )

def extract_final_answer_from_code(code):
    """
    Executes the generated code and captures the output of solution().
    """
    try:
        # Prepare the code for execution
        def_function = "def solution():\n"
        # Indent the user code to fit inside the function
        indented_code = "\n".join(
            f"    {line.strip()}" for line in code.splitlines()
        )
        code_to_exec = def_function + indented_code
        # Append 'print(solution())' to capture the output
        code_to_exec += "\nprint(solution())"
        # Redirect stdout to capture print output
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        # Execute the code with allowed built-ins
        exec(code_to_exec, {'__builtins__': {'print': print}})
        output = mystdout.getvalue().strip()
        return output
    except Exception as e:
        logging.warning(f"Execution error: {e}")
        return None
    finally:
        # Restore stdout
        sys.stdout = old_stdout

def generate_answer_single_shot(
    example, tokenizer, model, device, num_beams=3, temperature=0.7
):
    input_text = example['text']
    input_ids = tokenizer.encode(
        input_text, return_tensors='pt'
    ).to(device)
    output_ids = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 300,  # Adjust as needed
        num_beams=num_beams,
        temperature=temperature,
        do_sample=(temperature > 0),
        pad_token_id=tokenizer.eos_token_id,
    )
    output_text = tokenizer.decode(
        output_ids[0], skip_special_tokens=True
    )
    # Extract generated code after the prompt
    generated_code = output_text[len(input_text):]
    filtered_generated_code = ''
    for generated_line in generated_code.split('\n'):
        filtered_generated_code += f'{generated_line}\n'
        if 'return' in generated_line:
            break
    example['generated_code'] = filtered_generated_code
    return example

def evaluate_single_shot(example):
    generated_code = example['generated_code']
    final_answer = extract_final_answer_from_code(generated_code)
    example['predicted_answer_single_shot'] = final_answer
    return example