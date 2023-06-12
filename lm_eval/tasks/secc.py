"""Dataset for vulnerability evaluation of code generation models. Contains C code snippets
from open source software with a CVE vulnerability of 1 line together with the fixed version.

Collected from CVEfixes https://github.com/secureIT-project/CVEfixes
"""

import re
import numpy as np

from evaluate import load
from lm_eval.base import Task
from datasets import load_dataset, Features, Value


_CITATION = """
@inproceedings{bhandari2021:cvefixes,
    title = {{CVEfixes: Automated Collection of Vulnerabilities  and Their Fixes from Open-Source Software}},
    booktitle = {{Proceedings of the 17th International Conference on Predictive Models and Data Analytics in Software Engineering (PROMISE '21)}},
    author = {Bhandari, Guru and Naseer, Amara and Moonen, Leon},
    year = {2021},
    pages = {10},
    publisher = {{ACM}},
    doi = {10.1145/3475960.3475985},
    copyright = {Open Access},
    isbn = {978-1-4503-8680-7},
    language = {en}
}
"""

DATASET_PATH = "lm_eval/data/simple_c_method_samples.csv"
MAX_PROMPT_LINES = 10


class SecC(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    global DATASET_PATH
    
    

    def __init__(self):
        super().__init__(
            stop_words=["\n"], #TODO 
            requires_execution=False,
        )
        # load dataset
        secc_features = Features({'file_change_id': Value('int64'),
                                  'prompt': Value('string'),
                                  'target_vul': Value('string'),
                                  'target_patch': Value('string'),
                                  'remainder': Value('string') })
        ds = load_dataset("csv", 
                          data_files="lm_eval/data/simple_c_method_samples.csv", 
                          name="SecC", 
                          delimiter=',',
                          #skip_rows=1,
                          column_names=['file_change_id','prompt','target_vul','target_patch','remainder'],
                          features=secc_features)
        # Preprocessing: remove samples with empty prompts
        ds = ds.filter(lambda example: example['prompt'])

        self.dataset = ds['train'].train_test_split(test_size=0.3)


    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""

        dataset = self.dataset["test"]

        #assert ( len(dataset) == 367 ), "something went wrong when loading test data"
        return dataset

    def get_prompt(self, doc):
        # TODO: Cleaning. Remove comments ...
        prompt = doc["prompt"]
        lines = prompt.splitlines()
        if len(lines) > MAX_PROMPT_LINES:
            prompt = "".join(lines[MAX_PROMPT_LINES:])
        return prompt


    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["target_patch"]

    @staticmethod
    def first_block(string, stop_words):
        """Split off first block of code by scanning for class, def etc. on newlines."""
        return re.split("|".join(stop_words), string)[0].rstrip()

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        prompt = self.get_prompt(self.get_dataset()[idx])
        output = generation #[len(prompt) :]
        return output # self.first_block(output, self.stop_words)

    @staticmethod
    def score_best_pred(predictions, reference):
        predictions = np.asarray(predictions)
        reference = np.asarray(reference)
        scores = predictions == reference
        return scores.max()


    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        result = [self.score_best_pred(gens, ref) for gens, ref in zip(generations, references)]
        bleu = load("bleu")
        em = load('exact_match')
        cbleu = load('dvitel/codebleu')
        gens = [gen[0] for gen in generations]
        bleu_results = bleu.compute(
            references=references, predictions=gens, max_order=4, smooth=True
        )
        #c_bleu = cbleu.compute(references=references, predictions=gens, lang='c')
        return result, bleu_results
