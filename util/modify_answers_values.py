def move_answers_position(tokenized, answers):

	input_ids = tokenized['input_ids']
	offsets = tokenized['offset_mapping']
	mappings = tokenized['overflow_to_sample_mapping']

	new_answers = []
	all_inclusive = 0
	not_all_inclusive = 0
	start_inclusive = 0
	end_inclusive = 0
	for post_index, pre_index in enumerate(mappings):
		start_found = False
		end_found = False
		start_index_word = 0
		end_index_word = 0
		ans_start = answers[0][pre_index]
		ans_end = answers[1][pre_index]
		offset = offsets[post_index]
		input_id = input_ids[post_index]

		# the search starts after the question, so after the first 102 tag
		for i in range(input_id.index(102), len(offset)):
			of = offset[i]
			range_offset = list(range(of[0], of[1]+1))

			if ans_start in range_offset:
				# save the word index i as the answer start
				start_found = True
				start_index_word = i

			if ans_end in range_offset:
				# save the word index i as the answer end
				end_found = True
				end_index_word = i

			if start_found and end_found:
				break

		# the answer is completely included in this context -> no changes
		if start_found and end_found:
			all_inclusive += 1
			new_answers.append((start_index_word, end_index_word))
			continue
		else:
			# here's the problems
			not_all_inclusive += 1
			new_answers.append((0, 0))
			continue

	print("stats: (all, noth): {}".format((all_inclusive, not_all_inclusive)))
	return new_answers
