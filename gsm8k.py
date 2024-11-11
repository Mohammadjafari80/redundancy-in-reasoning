# gsm8k.py

import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)

class GSM8K:
    def __init__(self, split, include_answer=True, include_reasoning=True):
        self.split = split
        self.include_answer = include_answer
        self.include_reasoning = include_reasoning
        self.dataset = self.load_dataset()

    def process_example(self, example):
        question = example['question']
        answer = example['answer']
        # Extract the reasoning steps and the final answer
        answer_delim = "#### "
        if answer_delim in answer:
            reasoning = answer.split(answer_delim)[0].strip()
            final_answer = answer.split(answer_delim)[-1].strip()
        else:
            reasoning = answer.strip()
            final_answer = ''
        # Create the code solution
        if self.include_answer:
            if self.include_reasoning:
                # Include reasoning as comments in the code
                reasoning_lines = reasoning.split('\n')
                reasoning_code = '\n'.join([f"    # {line}" for line in reasoning_lines])
                code_solution = (
                    f"def solution():\n"
                    f"    \"\"\"{question}\"\"\"\n"
                    f"{reasoning_code}\n"
                    f"    return {repr(final_answer)}"
                )
            else:
                # Do not include reasoning; directly return final_answer
                code_solution = (
                    f"def solution():\n"
                    f"    \"\"\"{question}\"\"\"\n"
                    f"    return {repr(final_answer)}"
                )
        else:
            # Do not include the answer at all
            code_solution = f"def solution():\n    \"\"\"{question}\"\"\""
        # Construct the input text
        input_text = f"Q: {question}\n\n# solution in Python:\n\n{code_solution}\n\n"
        return {
            'text': input_text,
            'final_answer': final_answer,
            'question': question,
            'answer': answer,
        }

    def load_dataset(self):
        # Load the GSM8K dataset with 'main' config
        dataset = load_dataset('gsm8k', 'main', split=self.split)
        # Process the dataset
        dataset = dataset.map(self.process_example)
        return dataset
