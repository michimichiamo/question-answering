import numpy as np
import torch
from transformers import DistilBertModel
from torchmetrics import Accuracy, AveragePrecision, F1

class QA(torch.nn.Module):

    def __init__(self, hidden_size=768, num_labels=2, dropout_rate=0.5):
        super(QA, self).__init__()
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Parameters
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        # Layers
        #self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased-distilled-squad')
        self.transformers = DistilBertModel.from_pretrained('distilbert-base-cased-distilled-squad').to(self.device)
        self.transformers.requires_grad_(False)
        self.dropout = torch.nn.Dropout(dropout_rate)
        #self.extra_linear = torch.nn.Linear(self.hidden_size, self.hidden_size)
        #self.extra_linear_tanh = torch.nn.Tanh()
        self.dense = torch.nn.Linear(self.hidden_size, self.num_labels, device=self.device, dtype=torch.float32)

    def forward(self, inputs):
        # Unpack inputs
        input_ids, attention_mask = inputs

        # Put to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Transformers
        transformed = self.transformers(input_ids=input_ids, attention_mask=attention_mask)
        # Dropout
        dropped = self.dropout(transformed[0])
        # Obtain logits
        logits = self.dense(dropped) #(None, seq_len, hidden_size)*(hidden_size, 2)=(None, seq_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)    #(None, seq_len, 1), (None, seq_len, 1)
        start_logits = start_logits.squeeze(-1)  #(None, seq_len)
        end_logits = end_logits.squeeze(-1)    #(None, seq_len)
        # --- 4) Prepare output tuple
        outputs = (start_logits, end_logits)

        return outputs


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ids, input_ids, attention_masks, answer_starts, answer_ends):
        'Initialization'
        self.ids = ids
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.answer_starts = answer_starts
        self.answer_ends = answer_ends

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.input_ids)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.ids[index]
        input_id = self.input_ids[index]
        attention_mask = self.attention_masks[index]
        answer_start = self.answer_starts[index]
        answer_end = self.answer_ends[index]

        # Pack input and output
        X = (ID, input_id, attention_mask)
        y = (answer_start, answer_end)

        return X, y

def read_npz(path='./data/tokenized/', split=None, task='QA', return_qam=False):
 assert split in ['train', 'val']
 filename = path+split+'.npz'
 data = np.load(filename)

 if task == 'QA':
    ids = data['id']
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    answer_start = data['answer_start']
    answer_end = data['answer_end']

    return ids, input_ids, attention_mask, answer_start, answer_end

 elif task == 'QG':
     ids = data['id']
     context_input_ids = data['context_input_ids']
     context_attention_mask = data['context_attention_mask']
     question_input_ids = data['question_input_ids']
     question_attention_mask = data['question_attention_mask']

     # Return Question Attention mask, if required
     if return_qam:
         return ids, context_input_ids, context_attention_mask, question_input_ids, question_attention_mask
     return ids, context_input_ids, context_attention_mask, question_input_ids

 else:
     print('Task not recognized')
     raise NotImplementedError


def define_metrics(model):
    # Define scores and send to device
    f1_score = F1(num_classes=model.transformers.config.max_position_embeddings, mdmc_average='global')
    f1_score = f1_score.to(model.device)

    def exact_match(predictions, targets):
      sum_exact = 0
      for index, (p, t) in enumerate(zip(predictions, targets)):
        if torch.equal(torch.argmax(p), t):
          sum_exact += 1
      return sum_exact/predictions.size(dim=0)

    ## TODO
    ## IoU (Intersection over Union)

    metrics = {
    	'F1' : f1_score,
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
    exact_match = metrics['ExactMatch']

    # Get F1 scores
    f1_start = f1_score(start_preds, start_target)
    f1_end = f1_score(end_preds, end_target)
    f1 = (f1_start + f1_end)/2
    f1 = f1.to('cpu')

    # Get Exact Match scores
    em_start = exact_match(start_preds, start_target)
    em_end = exact_match(end_preds, end_target)
    em = (em_start + em_end)/2

    print('Evaluation completed.')
    print(f'F1: {f1}, Precision: {avg}, Accuracy: {acc}, Exact Match: {em}')

    return f1, em
