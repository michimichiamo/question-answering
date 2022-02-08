import json
import pandas as pd
import random
import numpy as np
from transformers import T5TokenizerFast


def tokenize(df, max_length=512, doc_stride=256, verbose=1):

	if verbose>0:
		print('Loading data...')
	# Extract data
	ids = df['id'].to_numpy()
	questions = df['question'].to_list()
	contexts = df['context'].to_list()
	answer_texts = df['answer_text'].to_list()
	answer_indices = np.array(list(zip(df['answer_start'].to_list(), df['answer_end'].to_list())))
	
	del df

	if verbose>1:
		print('Original sample:')
		print(f'question: {questions[0]}')
		print(f'context: {contexts[0]}')
		print(f'answer: {answer_indices[0]}')


	## Load tokenizer
	tokenizer = T5TokenizerFast.from_pretrained('t5-base')

	# Tokenization
	if verbose>0:
		print('Tokenization...(should take about 30 seconds)')
  
	tokenized_questions = tokenizer(
		questions,
		max_length=max_length,
		truncation=True,
		return_overflowing_tokens=True,
		return_offsets_mapping=True,
		stride=doc_stride,
		return_attention_mask=True,
		padding='max_length'
    )

	tokenized_contexts = tokenizer(
		answer_texts,
		contexts,
		max_length=max_length,
		truncation=True,
		return_overflowing_tokens=True,
		return_offsets_mapping=True,
		stride=doc_stride,
		return_attention_mask=True,
		padding='max_length'
	)

	del questions, contexts

	if verbose>1:
		print('Tokenized sample:')
		print(tokenized_questions['input_ids'][0])
		print(tokenized_contexts['input_ids'][0])

	# Create output DataFrame
	df = pd.DataFrame()
	# Store tokenization results
	df['context_input_ids'] = tokenized_contexts['input_ids']
	df['context_attention_mask'] = tokenized_contexts['attention_mask']
	#df['question_input_ids'] = tokenized_questions['input_ids']
	#df['question_attention_mask'] = tokenized_questions['attention_mask']

	df['offset_mapping'] = tokenized_contexts['offset_mapping']
	df['overflow_to_sample_mapping'] = tokenized_contexts['overflow_to_sample_mapping']
    # Store questions indexing on mapping
	df['question_input_ids'] = [list(item) for item in np.array(tokenized_questions['input_ids'])[df['overflow_to_sample_mapping'].values]]
	df['question_attention_mask'] = [list(item) for item in np.array(tokenized_questions['attention_mask'])[df['overflow_to_sample_mapping'].values]]
	# Store original answers (to be fixed), indexing on mapping
	df[['answer_start','answer_end']] = answer_indices[df['overflow_to_sample_mapping'].values]
	df['id'] = ids[df['overflow_to_sample_mapping'].values]

	if verbose>0:
		print('Done.')

	return df


def fix_answers(df):
	def helper(row):
	    # Extract context
	    context = row['offset_mapping']
	    # Extract answer
	    answer_start = row['answer_start']
	    answer_end = row['answer_end']
	    # Iterate over contexts to assign answer's word position
	    start_found, end_found = False, False
	    for idx, (start, end) in enumerate(context):
	        context_range = range(start, end+1)
	    
	        if start_found and end_found:
	            break
	        else:
	            if (not start_found) and (answer_start in context_range):
	                start_found = True
	                answer_start = idx
	            if (not end_found) and (answer_end in context_range):
	                end_found = True
	                answer_end = idx
	            
	    if not(start_found and end_found):
	        answer_start, answer_end = (0,0)
	    
	    return answer_start, answer_end

	df[['answer_start','answer_end']] = df.apply(helper, axis=1, result_type="expand")
