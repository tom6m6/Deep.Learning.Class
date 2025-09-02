import torch
import torch.nn as nn
import torch.nn.functional as F
from BertModel import *

class BertModel(nn.Module):
    """Construct a 'BERT' model to rank relations for each step"""
    def __init__(self, config, device = None):
        super(BertModel, self).__init__()
        self.hidden_size = config.hidden_size
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.classifier = nn.Linear(self.hidden_size, 2)
        self.device = device
        self.padding_id = 0

    def forward(self, sequence):


        sequence_mask = torch.eq(sequence, self.padding_id).type(torch.FloatTensor)
        sequence_mask = sequence_mask.to(self.device) if self.device else sequence_mask
        sequence_mask = sequence_mask
        sequence_mask = -1.e10 * sequence_mask

        sequence_emb = self.embeddings(sequence)
        sequence_enc = self.encoder(sequence_emb,
                                    sequence_mask,
                                    output_all_encoded_layers = False)[-1]
        sequence_pool = self.pooler(sequence_enc)

        logits = self.classifier(sequence_pool)
        return logits