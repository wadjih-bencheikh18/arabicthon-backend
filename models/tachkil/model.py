import torch.nn as nn
import torch.nn.functional as F

class TachkilLstmModel(nn.Module):
    def __init__(
        self,
        emb_dim,
        vocab_size,
        output_size,
    ):
        super(TachkilLstmModel, self).__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size

        self.embdding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=0)
        self.pos_embdding = nn.Embedding(
            num_embeddings=15, embedding_dim=emb_dim, padding_idx=0)

        self.lstm_layer_1 = nn.LSTM(
            input_size=emb_dim,
            hidden_size=256,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.lstm_layer_2 = nn.LSTM(
            input_size=self.lstm_layer_1.hidden_size * 2,
            hidden_size=256,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.lstm_layer_3 = nn.LSTM(
            input_size=self.lstm_layer_2.hidden_size * 2,
            hidden_size=256,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.fc1 = nn.Linear(self.lstm_layer_3.hidden_size * 2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, src):

        emb = self.embdding(src)

        lstm_1_seq, _ = self.lstm_layer_1(emb)
        lstm_2_seq, _ = self.lstm_layer_2(lstm_1_seq)
        lstm_3_seq, _ = self.lstm_layer_3(lstm_2_seq)

        out = self.dropout1(F.relu(self.fc1(lstm_3_seq)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc3(out)

        return out
