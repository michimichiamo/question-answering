import torch
from transformers import DistilBertTokenizerFast

def evaluate_model(model, dataloader, weights_path=resdir+'/weights/weights20220119_39')
    model.dense.load_state_dict(torch.load(weights_path))
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased-distilled-squad')

    for inputs, targets in dataloader:
        hits = 0 # Correct prediction
        pred_ss_true = 0 # prediction contained in target
        true_ss_pred = 0 # viceversa
        # Unpack targets
        start_lbl, end_lbl = targets
        start_lbl = start_lbl.argmax(axis=1)
        end_lbl = end_lbl.argmax(axis=1)
        # Get predictions
        start_logits, end_logits = model.forward(inputs)
        start_preds = start_logits.argmax(axis=1).to('cpu')
        end_preds = end_logits.argmax(axis=1).to('cpu')
        # Iterate over batch
        for sample in range(256):
            x_0 = inputs[0][sample]
            # Decode target and prediction
            y_true = tokenizer.decode(x_0[start_lbl[sample]:end_lbl[sample]+1])
            y_pred = tokenizer.decode(x_0[start_preds[sample]:end_preds[sample]+1])
            # Count
            if y_true == y_pred:
                hits+=1
            elif y_pred in y_true:
                pred_ss_true+=1
            elif y_true in y_pred:
                true_ss_pred+=1
        print(hits, pred_ss_true, true_ss_pred)