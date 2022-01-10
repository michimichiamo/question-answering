def move_answers_position(offsets, answer):
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

    for i, of in enumerate(offset):
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

      # if the answer isn't included at all -> put (0,0) as answer index
      if not start_founded and not end_founded:
        not_all_inclusive += 1
        new_answers.append((0, 0))
        continue

      # if the start answer is included and not the end -> answer_start remains, answer end become the end
      if start_founded and not end_founded:
        print("Start inclusive")
        start_inclusive += 1
        print((start_index_word, end_index_word))
        new_answers.append((start_index_word, len(offset)))
        continue

      # if the start answer isn't included and the end is -> answer_start 0, answer end takes word index
      if not start_founded and end_founded:
        print("End inclusive")
        print(index)
        print(offset)
        print(ans_start)
        print(ans_end)
        end_inclusive += 1
        print((start_index_word, end_index_word))
        new_answers.append((0, end_index_word))
        continue

  print("stats: (all, noth, start, end): {}".format((all_inclusive, not_all_inclusive, start_inclusive, end_inclusive)))
  return new_answers
