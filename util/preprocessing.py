import json
import pandas as pd
import random
import numpy as np
from transformers import DistilBertTokenizerFast


"""
Per come dobbiamo dare i dati alla NN cioè concatenando context e question, per il fatto che i prof vogliono come output un dict {id_question, testo}
per avere i dati comodi la lista è formata da:
[id, title, context, question, answer_start, answer_end]
in tale modo per il training i dati sono [context, question] e il target è [answer_start, answer_end]
"""

def read_from_json(path='./SQUAD MATERIAL/training_set.json'):

    with open(path, 'r') as file:
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
    df = df[['id', 'title', 'context', 'question', 'answer_start', 'answer_end', 'answer_text']]
    
    return df


def tokenize(df, max_length=512, doc_stride=256, verbose=1):

	if verbose>0:
		print('Loading data...')
	# Extract data
	ids = df['id'].to_numpy()
	questions = df['question'].to_list()
	contexts = df['context'].to_list()
	answers = np.array(list(zip(df['answer_start'].to_list(), df['answer_end'].to_list())))
	
	del df

	if verbose>1:
		print('Original sample:')
		print(f'question: {questions[0]}')
		print(f'context: {contexts[0]}')
		print(f'answer: {answers[0]}')


	## Load tokenizer
	tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased-distilled-squad')

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
	# Store original answers (to be fixed), indexing on mapping
	df[['answer_start','answer_end']] = answers[df['overflow_to_sample_mapping'].values]
	df['id'] = ids[df['overflow_to_sample_mapping'].values]

	if verbose>0:
		print('Done.')

	return df


def fix_answers(df):
	def helper(row):
		# Extract question length
		question_length = len(row['offset_mapping'][:row['input_ids'].index(102)])
		# Extract context: from first 'CLS' encountered, filter out (0,0)s (to avoid padding)
		context = row['offset_mapping'][row['input_ids'].index(102):]
		#context_nonzero = list(filter(lambda x: x!=(0,0), context))
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
					answer_start = idx + question_length
				if (not end_found) and (answer_end in context_range):
					end_found = True
					answer_end = idx + question_length
			    
		if not(start_found and end_found):
			answer_start, answer_end = (0,0)
		
		return answer_start, answer_end

	df[['answer_start','answer_end']] = df.apply(helper, axis=1, result_type="expand")


def one_hot_answers(df):
	out_df = df.drop(['answer_start', 'answer_end'], axis=1).copy()
	# Prepare output
	answer_start = np.zeros(shape=(len(df), 512), dtype='int32')
	answer_end = np.zeros(shape=(len(df), 512), dtype='int32')

	# Iterate over DataFrame
	for row, (start_value, end_value) in enumerate(zip(df['answer_start'], df['answer_end'])):
	    answer_start[row][start_value] = 1
	    answer_end[row][end_value] = 1
	    
	# Replace columns in DataFrame
	out_df['answer_start'] = list(answer_start)
	out_df['answer_end'] = list(answer_end)

	return out_df

#def read_from_json(path='training_dataset.json', validation_set_perc = 0.0, limit_dataset=0):
#	'''
#				@param path: path to dataset file
#				@param validation_set_perc: inserire la percentuale in [0,1] del dataset per la validazione, se 0 non viene creato il validation set
#				@param limit_dataset: if necessary to not use all the dataset set a limit of titles to include
#	'''
#	dataset = None
#	with open(path) as file:
#		dataset = pd.read_json(file)
#
#	temp_dataset = []
#	training_dataset = []
#	validation_dataset = []
#
#	titles = []
#	i = 0
#	for index, row in dataset.iterrows():
#		#print(row['data'])
#		data = row['data']
#		title = data['title']
#		titles.append(title)
#		for ps in data['paragraphs']:
#			#print(ps)
#			context = ps['context']
#			for qas in ps['qas']:
#				id = qas['id']
#				question = qas['question']
#				for ans in qas['answers']:
#					training_dataset.append([id, title, context, question, ans['answer_start'], ans['answer_start']+len(ans['text'])])
#		if limit_dataset != 0 and i >= limit_dataset:
#				break
#
#	if validation_set_perc > 0.0:
#		titles_validation = random.choices(titles, k=int(np.ceil(len(titles)*validation_set_perc/100)))
#		for data in training_dataset:
#			if data[1] in titles_validation:
#				validation_dataset.append(data)
#			else:
#				temp_dataset.append(data)
#		training_dataset = temp_dataset
#
#	return training_dataset, validation_dataset


#def move_answers_position(tokenized, answers):
#
#	input_ids = tokenized['input_ids']
#	offsets = tokenized['offset_mapping']
#	mappings = tokenized['overflow_to_sample_mapping']
#
#	new_answers = []
#	all_inclusive = 0
#	not_all_inclusive = 0
#	start_inclusive = 0
#	end_inclusive = 0
#	for post_index, pre_index in enumerate(mappings):
#		start_found = False
#		end_found = False
#		start_index_word = 0
#		end_index_word = 0
#		ans_start = answers[pre_index][0]
#		ans_end = answers[pre_index][1]
#		offset = offsets[post_index]
#		input_id = input_ids[post_index]
#
#		# the search starts after the question, so after the first 102 tag
#		for i in range(input_id.index(102), len(offset)):
#			of = offset[i]
#			range_offset = list(range(of[0], of[1]+1))
#
#			if ans_start in range_offset:
#				# save the word index i as the answer start
#				start_found = True
#				start_index_word = i
#
#			if ans_end in range_offset:
#				# save the word index i as the answer end
#				end_found = True
#				end_index_word = i
#
#			if start_found and end_found:
#				break
#
#		# the answer is completely included in this context -> no changes
#		if start_found and end_found:
#			all_inclusive += 1
#			new_answers.append((start_index_word, end_index_word))
#			continue
#		else:
#			# here's the problems
#			not_all_inclusive += 1
#			new_answers.append((0, 0))
#			continue
#
#	print("stats: (all, noth): {}".format((all_inclusive, not_all_inclusive)))
#	return new_answers

