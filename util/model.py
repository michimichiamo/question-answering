	# Define scores and send to device
	f1_score = F1(num_classes=model.transformers.config.max_position_embeddings, mdmc_average='global')
	f1_score = f1_score.to(model.device)
	average_precision = AveragePrecision(pos_label=1, num_classes=model.transformers.config.max_position_embeddings)
	average_precision = average_precision.to(model.device)
	accuracy = Accuracy(mdmc_average='global', num_classes=model.transformers.config.max_position_embeddings)
	accuracy = accuracy.to(model.device)

	def exact_match(predictions, targets):
		sum_exact = 0
		for index, (p, t) in enumerate(zip(predictions, targets)):
			if torch.equal(torch.argmax(p), torch.argmax(t)):
				sum_exact += 1
		return sum_exact/index

	## TODO
	## IoU (Intersection over Union)

	metrics = {
    	'F1' : f1_score,
    	'Precision' : average_precision,
    	'Accuracy' : accuracy,
      'ExactMatch': exact_match
	}

	return metrics

def evaluate(model, inputs, targets, metrics):
    # Set evaluation mode
    model.eval()
    # Obtain predictions
    start_preds, end_preds = model.forward(inputs)
    # Unpack targets and send to device
    start_target, end_target = targets
    start_target = start_target.to(model.device)
    end_target = end_target.to(model.device)
    
#    # Extract IntTensors for predictions
#    start_preds, end_preds = torch.zeros_like(start_model, dtype=torch.int16), torch.zeros_like(end_model, dtype=torch.int16)
#    start_preds[torch.tensor(range(start_model.size()[0])), torch.argmax(start_model, axis=1)] = 1
#    end_preds[torch.tensor(range(end_model.size()[0])), torch.argmax(end_model, axis=1)] = 1

    # Send predictions to device
    start_preds = start_preds.to(model.device)
    end_preds = end_preds.to(model.device)

    f1_score = metrics['F1']
    average_precision = metrics['Precision']
    accuracy = metrics['Accuracy']
    exact_match = metrics['ExactMatch']

    # Get F1 scores
    f1_start = f1_score(start_preds, start_target)
    f1_end = f1_score(end_preds, end_target)
    f1 = (f1_start + f1_end)/2
    f1 = f1.to('cpu')
    
    # Get Average Precision scores
    avg_start = average_precision(start_preds, start_target)
    avg_end = average_precision(end_preds, end_target)
    avg = (avg_start + avg_end)/2
    avg = avg.to('cpu')

    # Get Accuracy scores
    acc_start = accuracy(start_preds, start_target)
    acc_end = accuracy(end_preds, end_target)
    acc = (acc_start + acc_end)/2
    acc = acc.to('cpu')

    # Get Accuracy scores
    em_start = exact_match(start_preds, start_target)
    em_end = exact_match(end_preds, end_target)
    em = (em_start + em_end)/2

    print('Evaluation completed.')
    print(f'F1: {f1}, Precision: {avg}, Accuracy: {acc}, Exact Match: {em}')
                
    return f1, avg, acc