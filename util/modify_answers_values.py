def move_answers_position(input_ids, offsets, answer):
  new_answers = []
  all_inclusive = 0
  not_all_inclusive = 0
  start_inclusive = 0
  end_inclusive = 0
  for index in range(0, len(offsets)):
    start_founded = False
    end_founded = False
    start_index_word = 0
    end_index_word = 0
    ans_start = answer[0][index]
    ans_end = answer[1][index]
    offset = offsets[index]
    input_id = input_ids[index]

    # the search starts after the question, so after the first 102 tag
    for i in range(input_id.index(102), len(offset)):
      of = offset[i]
      range_offset = list(range(of[0], of[1]+1))

      if ans_start in range_offset:
        # save the word index i as the answer start
        start_founded = True
        start_index_word = i

      if ans_end in range_offset:
        # save the word index i as the answer end
        end_founded = True
        end_index_word = i

      if start_founded and end_founded:
        break

    # the answer is completely included in this context -> no changes
    if start_founded and end_founded:
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
