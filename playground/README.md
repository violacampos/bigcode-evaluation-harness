# Models

#### Candidates from huggingface:

- NinedayWang/PolyCoder-2.7B
- NinedayWang/PolyCoder-0.4B


- facebook/incoder-1B
- facebook/incoder-6B

- EleutherAI/gpt-j-6B (see https://discuss.huggingface.co/t/closest-model-available-to-openais-codex-github-copilot-for-code-completion/15224/2)

- Salesforce/codet5-large-ntp-py
- Salesforce/codet5-large

- Salesforce/codegen-6B-nl
- Salesforce/codegen-6B-multi


- https://huggingface.co/spaces/bigcode/santacoder-demo
- https://huggingface.co/spaces/lvwerra/codeparrot-generation

#### Others

- UnixCoder
- Llama?
- text-davinci-003 (openai-gpt3)
- code-davinci-002
- CompCoder? (Compilable Neural Code Generation with Compiler Feedback)
- PPOCoder https://github.com/reddy-lab-code-research/PPOCoder
- google/flan-t5-base (auch large, small, xl) -> not specifically trained on code
- Grammformer https://arxiv.org/abs/2106.10158

## logs
```
accelerate launch  main.py   --model 'facebook/incoder-6B'   --max_length_generation 128   --tasks mbpp   --temperature 0.1   --n_samples 15   --batch_size 10   --allow_code_execution --save_generations --limit 30
```

  "mbpp": {
    "pass@1": 0.04666666666666666,
    "pass@10": 0.2
  }

```
accelerate launch  main.py   --model 'facebook/incoder-6B'   --max_length_generation 512   --tasks mbpp   --temperature 0.1   --n_samples 15   --batch_size 10   --allow_code_execution --save_generations --limit 30
```

"mbpp": {
    "pass@1": 0.04666666666666667,
    "pass@10": 0.16666666666666666
  }

accelerate launch  main.py   --model 'facebook/incoder-1B'   --max_length_generation 128   --tasks mbpp   --temperature 0.1   --n_samples 3   --batch_size 5   --allow_code_execution --save_generations --limit 30

{
  "mbpp": {
    "pass@1": 0.0
  },
  "config": {
    "model": "facebook/incoder-1B",
    "temperature": 0.1,
    "n_samples": 3
  }
}