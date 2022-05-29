import torch.nn as nn
import torch.nn.functional as F


class LstmModel(nn.Module):
    def __init__(
        self,
        emb_dim,
        vocab_size,
        output_size,
        lstm_dim=128,
    ):
        super(LstmModel, self).__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size

        self.embdding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=0)
        self.pos_embdding = nn.Embedding(
            num_embeddings=15, embedding_dim=emb_dim, padding_idx=0)

        self.lstm_layer_1 = nn.LSTM(
            input_size=emb_dim,
            hidden_size=lstm_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.lstm_layer_2 = nn.LSTM(
            input_size=self.lstm_layer_1.hidden_size * 2,
            hidden_size=lstm_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.lstm_layer_3 = nn.LSTM(
            input_size=self.lstm_layer_2.hidden_size * 2,
            hidden_size=lstm_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.fc1 = nn.Linear(self.lstm_layer_3.hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, output_size)

        self.dropout1 = nn.Dropout(0.5)

    def forward(self, src):

        emb = self.embdding(src)

        lstm_1_seq, _ = self.lstm_layer_1(emb)
        lstm_2_seq, _ = self.lstm_layer_2(lstm_1_seq)
        lstm_3_seq, _ = self.lstm_layer_3(lstm_2_seq)

        lstm_out = lstm_3_seq[:, -1, :]

        features = lstm_out

        out = self.dropout1(F.relu(self.fc1(features)))
        out = self.fc2(out)

        return out
