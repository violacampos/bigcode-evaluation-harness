import torch
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, Features, Value

checkpoint = "Salesforce/codet5p-2b"
device = "cuda:7" # for GPU usage or "cpu" for CPU usage
commit = "0d030d0077331c69e011d3401f783254b8201330"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                              torch_dtype=torch.float16,
                                              trust_remote_code=True,
                                              revision=commit).to(device)

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
ds = ds['train'].train_test_split(test_size=0.3)

encodings = tokenizer(ds["test"]["prompt"], return_tensors="pt")

max_length = model.config.n_positions
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())