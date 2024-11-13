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
    examples, tokenizer, model, device, num_beams=3, temperature=0.7
):
    input_texts = examples['text']
    # Tokenize inputs
    inputs = tokenizer(
        input_texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512,  # Adjust as needed
    ).to(device)
    
    with torch.no_grad():
        if device.type == 'cuda':
            # with torch.amp.autocast('cuda'):
            output_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],  # Try commenting out
                max_length=inputs['input_ids'].shape[1] + 512,  # Total length
                num_beams=num_beams,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            output_ids = model.generate(
                inputs['input_ids'],
                # attention_mask=inputs['attention_mask'],  # Try commenting out
                max_length=inputs['input_ids'].shape[1] + 512,
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

def evaluate_single_shot(example):
    generated_code = example['generated_code']
    final_answer = extract_final_answer_from_code(generated_code)
    example['predicted_answer_single_shot'] = final_answer
    return example