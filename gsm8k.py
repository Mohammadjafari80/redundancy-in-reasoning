# gsm8k.py

import logging
import random
import re
import json
from datasets import load_dataset

logger = logging.getLogger(__name__)

class GSM8K:
    def __init__(self, split, include_answer=True, include_reasoning=True, p=0.0, few_shot=False, num_shots=8, seed=None, cot=False, template="code"):
        self.split = split
        self.include_answer = include_answer
        self.include_reasoning = include_reasoning
        self.p = p
        self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)
        
        assert not few_shot or split == 'test'
        
        self.few_shot = few_shot # If true, each example is processed with num_shots examples before it
        self.num_shots = num_shots
        self.cot = cot
        self.template = template
        self.examples = None
        
        self.dataset = self.load_dataset()

    def format_example(self, question, solution, answer):
        example = ''
        if self.template == 'code':
            example = f'Question: {question}\n\n# solution in Python:\n\ndef solution():\n    """{question}"""\n'
            
            if solution is not None:
                example += f'{solution}\n\n'
                
        elif self.template == 'qa':
            example = f"Question: {question}\nSolution: "
            
            # if self.cot:
            #     example += "Let's break it down step by step. "
            
            # example += '\n'
            
            if solution is not None:
                def remove_placeholders(text):
                    import re
                    # Regex to match <<anything>>
                    cleaned_text = re.sub(r'<<.*?>>', '', text)
                    return cleaned_text

                solution = '. '.join(solution.split('\n'))
                solution = remove_placeholders(solution)
                example += f"{solution}.\n"
            
            example = example.replace('..', '.')
            
            if answer is not None:
                example += f"#### The final answer is {answer}\n\n"
                
        else:
            raise ValueError('Format Not Implemented')
        
        return example
                
    def process_example(self, example, index):
        # Set seed based on self.seed and index for reproducibility
        if self.seed is not None:
            random.seed(self.seed + index)

        question = example['question']

        # With probability self.p, inject a random sentence
        if random.random() < self.p:
            # Split the question into sentences
            question_sentences = re.split(r'(?<=[.!?]) +', question)
            # Extract sentences that do not contain question marks
            current_non_question_sentences = [s for s in question_sentences if '?' not in s]

            # Exclude the last sentence if it usually asks the question
            if '?' in question_sentences[-1]:
                body_sentences = question_sentences[:-1]
                last_sentence = question_sentences[-1]
            else:
                body_sentences = question_sentences
                last_sentence = ''

            # Create a set of sentences from the current question to exclude
            current_sentences_set = set(current_non_question_sentences)

            # Create a list of possible sentences to inject (excluding current example's sentences)
            possible_sentences = [s for s in self.other_sentences if s not in current_sentences_set]

            if possible_sentences:
                # Choose a random sentence to inject
                injected_sentence = random.choice(possible_sentences)

                # Choose a random position to insert (not at the end)
                if len(body_sentences) > 0:
                    insert_position = random.randint(0, len(body_sentences) - 1)
                    body_sentences.insert(insert_position, injected_sentence)
                else:
                    # If no body sentences, insert before the last sentence
                    body_sentences = [injected_sentence]

                # Reconstruct the question
                if last_sentence:
                    question = ' '.join(body_sentences + [last_sentence])
                else:
                    question = ' '.join(body_sentences)
            else:
                # No possible sentences to inject; proceed without injection
                pass

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
                input_text = self.format_example(question, reasoning, final_answer)
            else:
                input_text = self.format_example(question, None, final_answer)
        else:
            input_text = self.format_example(question, None, None)

        
        if self.few_shot:
            input_text = self.few_shot_prompt + input_text
            
        return {
            'text': input_text,
            'final_answer': final_answer,
            'question': question,
            'answer': answer,
        }

    def load_dataset(self):
        # Load the GSM8K dataset with the specified split
        dataset = load_dataset('gsm8k', 'main', split=self.split)
        # Collect sentences from questions (excluding those containing question marks)
        sentences = []
        for example in dataset:
            question_text = example['question']
            # Split question into sentences
            question_sentences = re.split(r'(?<=[.!?]) +', question_text)
            # Exclude sentences that contain question marks
            non_question_sentences = [s for s in question_sentences if '?' not in s]
            sentences.extend(non_question_sentences)
        self.other_sentences = sentences
        
        if self.few_shot:
            self.few_shot_prompt = self.build_prompt()

        dataset = dataset.map(self.process_example, with_indices=True)
        return dataset

    def fewshot_examples_code(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        with open(
            "resources/gsm8k_few_shot_prompts.json",
            "r",
        ) as file:
            examples = json.load(file)
        
        new_examples = []
        for question, solution in zip(
            examples["questions"][:self.num_shots], examples["solutions"][:self.num_shots]
        ):
           new_examples.append(dict(
                question=question,
                solution=solution,
                final_answer=None
            ))

            
        return new_examples
    
    def fewshot_examples_qa(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        
        trainset = load_dataset('gsm8k', 'main', split=self.split)
        
        new_examples = []
        
        for example in trainset:
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
                
            new_examples.append(dict(
                question=question,
                solution=reasoning,
                final_answer=final_answer
            ))

            
        return new_examples

    def make_prompts(self):
        """Builds the prompt for the LM to generate from."""
        if self.template == 'code':
            examples = self.fewshot_examples_code()
        elif self.template == 'qa':
            examples = self.fewshot_examples_qa()
    
        self.examples = examples        
    
    def build_prompt(self):
        
        if self.examples is None:
            self.make_prompts()
        
        random.seed(42)
        
        prompt = '' 
        for qna in random.sample(self.examples, self.num_shots):
            prompt += self.format_example(qna['question'], qna['solution'], qna['final_answer'])
            
        return prompt
         
    def view_example(self, index):
        example = self.dataset[index]
        print("Question:")
        print(example['question'])
        print("\nProcessed Text:")
        print(example['text'])
        print("\nFinal Answer:")
        print(example['final_answer'])