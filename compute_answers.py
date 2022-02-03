import sys
import json
import pandas as pd
import random
import numpy as np
from transformers import DistilBertTokenizerFast
from util.model import QA
import torch

def read_from_json(file_path='./SQUAD MATERIAL/training_set.json'):

    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Load data into DataFrame
    df = pd.json_normalize(data['data'],
                           record_path=['paragraphs', 'qas', 'answers'],
                           meta=['title',
                                ['paragraphs', 'context'],
                                ['paragraphs', 'qas', 'question'],
                                ['paragraphs', 'qas', 'id']]
                          )
    # Rename columns
    mapper = {
        'paragraphs.context' : 'context',
        'paragraphs.qas.question' : 'question',
        'paragraphs.qas.id' : 'id',
        'text' : 'answer_text'
    }
    
    df.rename(mapper, axis=1, inplace=True)
    
    # Compute answer_end
    df['answer_end'] = df.apply(lambda x: x['answer_start']+len(x['answer_text']), axis=1)
    
    # Check consistency
    assert np.equal(
        df.apply(lambda x: x['context'][x['answer_start']:x['answer_end']], axis=1),
        df['answer_text']
    ).all()
    
    # Reorganize columns
    df = df[['id', 'title', 'context', 'question']][:100]
    
    return df


def tokenize(df, tokenizer, max_length=512, doc_stride=256, verbose=1):

	if verbose>0:
		print('Loading data...')
	# Extract data
	ids = df['id'].to_numpy()
	questions = df['question'].to_list()
	contexts = df['context'].to_list()
	
	del df

	if verbose>1:
		print('Original sample:')
		print(f'question: {questions[0]}')
		print(f'context: {contexts[0]}')


	# Tokenization
	if verbose>0:
		print('Tokenization...(should take about 30 seconds)')
	tokenized = tokenizer(
		questions,
		contexts,
		max_length=max_length,
		truncation="only_second",
		return_overflowing_tokens=True,
		return_offsets_mapping=True,
		stride=doc_stride,
		return_attention_mask=True,
		padding='max_length'
	)

	del questions, contexts

	if verbose>1:
		print('Tokenized sample:')
		print(tokenized['input_ids'][0])

	# Create output DataFrame
	df = pd.DataFrame()
	# Store tokenization results
	df['input_ids'] = tokenized['input_ids']
	df['attention_mask'] = tokenized['attention_mask']
	df['offset_mapping'] = tokenized['offset_mapping']
	df['overflow_to_sample_mapping'] = tokenized['overflow_to_sample_mapping']
	df['id'] = ids[df['overflow_to_sample_mapping'].values]

	if verbose>0:
		print('Done.')

	return df

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ids, input_ids, attention_masks):
        'Initialization'
        self.ids = ids
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.input_ids)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.ids[index]
        input_id = self.input_ids[index]
        attention_mask = self.attention_masks[index]

        # Pack input and output
        X = (ID, input_id, attention_mask)

        return X

def create_dataloader(df):

	# Load data
	ids = df['id'].values.astype(np.unicode_)
	input_ids = np.stack(df['input_ids']).astype('int32')
	attention_masks = np.stack(df['attention_mask']).astype('int32')

	dataset = Dataset(ids, input_ids, attention_masks)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, pin_memory=True)

	return dataloader

def compute_answers(model, loader, tokenizer):

	out = {}

	for i, X in enumerate(loader):
		# Unpack the input
		ID, input_id, attention_mask = X

		# Perform inference step
		start_outputs, end_outputs = model.forward((input_id, attention_mask))
		start_predictions = torch.argmax(start_outputs, axis=1)
		end_predictions = torch.argmax(end_outputs, axis=1)
	
		# Extract answer from context
		out[ID[index]] = tokenizer.decode(input_id[index][start:end+1])

		# Size limit
		if i > 4:
			break
	
	return out


if __name__ == '__main__':
	# Manage usage errors
	if len(sys.argv) != 2:
	    sys.exit('Usage: python3 compute_answers.py dataset.json')
	# Read JSON file
	file_path = sys.argv[1]
	data = read_from_json(file_path=file_path)

	## Load tokenizer
	tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased-distilled-squad')

	# Tokenization
	df = tokenize(data, tokenizer)

	# Create dataloader
	loader = create_dataloader(df)

	# Load model weights
	model = QA()
	model.dense.load_state_dict(torch.load(model_path))

	# Extract answers
	model_path = './data/model/best_model'
	out = compute_answers(model, loader, tokenizer)

	with open('predictions.txt', 'w') as f:
		print(out, file=f)










