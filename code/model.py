import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

from transformers import AutoModel


from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F



local_model_path = '/home/xxxxx/unixcoder-nine'
global_encoder_model = AutoModel.from_pretrained(local_model_path)
global_encoder_tokenizer = AutoTokenizer.from_pretrained(local_model_path)


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.global_encoder = global_encoder_model 

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8),
            num_layers=8
        )

        self.classifier = nn.Linear(config.hidden_size*3, 1)


    def forward(self, batch_input_ids, global_input_ids, labels=None):
        global_attention_mask = global_input_ids.ne(self.tokenizer.pad_token_id)
   
        global_cls = self.encoder(global_input_ids, attention_mask=global_attention_mask, output_hidden_states=True)
        global_cls = global_cls.hidden_states[-1][:, 0, :]




 
        flat_input_ids = batch_input_ids.view(-1, batch_input_ids.size(-1))
        flat_attention_mask = flat_input_ids.ne(self.tokenizer.pad_token_id)

   
        outputs = self.global_encoder(flat_input_ids, attention_mask=flat_attention_mask, output_hidden_states=True)
        flat_sentence_cls = outputs.hidden_states[-1][:, 0, :]
   
        sentence_cls = flat_sentence_cls.view(batch_input_ids.size(0), batch_input_ids.size(1), -1)
 

        sentence_means = sentence_cls.mean(dim=2)  


        min_sentence_indices = sentence_means.argmin(dim=1)  


        sentence_logits_all = torch.stack([sentence_cls[i, idx] for i, idx in enumerate(min_sentence_indices)]) 

        sentence_cls_T=sentence_cls.detach()
        transformer_output = self.transformer_encoder(sentence_cls_T)[:, -1, :]

 
        combined_cls = torch.cat([global_cls, sentence_logits_all,transformer_output], dim=1)

 

        final_predictions = self.classifier(combined_cls)
    

        final_predictions= torch.sigmoid(final_predictions)
  
  
        if labels is not None:
            labels = labels.float()
       
            loss = torch.log(final_predictions[:, 0] + 1e-10) * labels + torch.log((1 - final_predictions)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss

            loss=loss.mean()
            return loss, final_predictions
        else:
            return final_predictions
 
