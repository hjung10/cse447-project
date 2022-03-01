import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import os
from pathlib import Path
# from ast import literal_eval
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import math
from datasets import load_dataset
import nltk
from nltk import word_tokenize
from tqdm import tqdm
import json

torch.manual_seed(1)


def load_training_data():
    def build_vocab(train_data):
        # x = 'Happy New Yea That’s one small ste one giant leap for mankin'
        # x = set(x)
        # char_to_idx = {c: j for j, c in enumerate(x)}

        # ~ represents UNK character
        char_to_idx = {"'": 0, "<UNK>":1}
        for seq in train_data:
            for c in seq:
                if c not in char_to_idx:
                    char_to_idx[c] = len(char_to_idx)
        idx_to_char = {j: c for c, j in char_to_idx.items()}
        return char_to_idx, idx_to_char
    # your code here
    # this particular model doesn't train
    # adding more languages other than English to be a little more robust/generalizable
    top_languages = ['english', 'spanish', 'chinese_simplified', 'russian', 'hindi', 'japanese', "bengali"]
    lang_datasets = [load_dataset('csebuetnlp/xlsum', lang, split='train') for lang in top_languages]

    # manually reading data because load_dataset caused disk quota exceeded error
    """
    lang_datasets = []
    for lang in ['english']:
        path = Path('./XLSum_complete_v2.0/{}_train.jsonl'.format(lang))
        with open(path, 'rb') as f:
            json_list = [literal_eval(line.decode('utf8')) for line in f]
            lang_datasets.append(json_list)
    """

    sentences = []
    max_len = 10000
    # TODO: set this to be dataset.num_rows when we want to train on a larger set of data
    num_rows = 5000  # 2
    for dataset in lang_datasets:
        for i in range(num_rows):
            sentences.append(dataset[i]['text'][:max_len])

    char_to_idx, idx_to_char = build_vocab(sentences)

    input = []
    targets = []
    for i, sentence in enumerate(sentences):
        sentence = ''.join(sentence.split())
        # input & targets are on continuous list of integers
        input.extend([char_to_idx[c] for c in sentence[:-1]])
        targets.extend([char_to_idx[c] for c in sentence[1:]])

    input = torch.tensor(input)
    targets = torch.tensor(targets)

    return input, targets, char_to_idx, idx_to_char


def load_test_data(fname):
    # your code here
    data = []
    with open(fname) as f:
        for line in f:
            inp = line[:-1]  # the last character is a newline
            data.append(inp)
    return data


def write_pred(preds, fname):
    with open(fname, 'wt') as f:
        for p in preds:
            f.write('{}\n'.format(p))


def save(model, char_to_idx, idx_to_char, work_dir):
    torch.save(model.state_dict(), os.path.join(work_dir, 'model.checkpoint'))
    dictionaries = {
            'idx_to_char': idx_to_char,
            'char_to_idx': char_to_idx
            # 'unigrams': self.unigrams_context_freq,
            # 'bigrams': self.bigrams_context_freq,
            # 'trigrams': self.trigrams_context_freq
        }
    with open(os.path.join(work_dir, 'model.dictionary'), 'w') as output_json:
        json.dump(dictionaries, output_json)

def load(model, work_dir):
    # checkpoint = torch.load(work_dir, map_location='cpu')
    # model.load_state_dict(checkpoint)
    model.load_state_dict(torch.load(os.path.join(work_dir, 'model.checkpoint')))

def load_dictionary(work_dir):
    path = Path(os.path.join(work_dir, 'model.dictionary'))
    with open(path, 'rb') as f:
        data = json.load(f)

    idx_to_char = dict()
    char_to_idx = data["char_to_idx"]
    for k, v in data["idx_to_char"].items():
        idx_to_char[int(k)] = v 
    return idx_to_char, char_to_idx


class Text_Dataset(torch.utils.data.Dataset):
    def __init__(self, encoded_input, encoded_targets, sequence_len, batch_size):
        self.encoded_input = encoded_input
        self.encoded_targets = encoded_targets
        self.sequence_len = sequence_len
        self.batch_size = batch_size

        remainder = len(self.encoded_input) % self.batch_size
        self.input = torch.LongTensor(self.encoded_input[:len(self.encoded_input) - remainder])
        self.input = self.input.view(self.batch_size, -1)

        self.targets = torch.LongTensor(self.encoded_targets[:len(self.encoded_targets) - remainder])
        self.targets = self.targets.view(self.batch_size, -1)

        self.sequences_in_batch = math.ceil((self.input.shape[1] - 1) / self.sequence_len)

    def __getitem__(self, idx):
        batch_idx = idx % self.batch_size
        sequence_idx = idx // self.batch_size

        start_idx = sequence_idx * self.sequence_len
        input = self.input[batch_idx][start_idx: min(self.input.shape[1], start_idx + self.sequence_len + 1)]
        targets = self.targets[batch_idx][start_idx: min(self.targets.shape[1], start_idx + self.sequence_len + 1)]
        # return input[:-1], input[1:]
        item = {'input': input, 'targets': targets}
        return item

    def __len__(self):
        return self.batch_size * self.sequences_in_batch


class LSTMGenerator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size):
        super(LSTMGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_size = output_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        # lstm_out, _ = self.lstm(embeds.view(embeds.shape[0], 1, -1))
        # lstm_out, _ = self.gru(embeds.view(len(sentence), 1, -1))
        output = self.fc(lstm_out)
        # output = self.fc(lstm_out.view(len(sentence), -1))
        # output_scores = F.log_softmax(output, dim=1)
        output = output.reshape(-1, self.output_size)
        return output


def train(train_dataloader, model, loss_fn, optimizer, epochs=1):  # train_input, train_targets
    train_correct = 0
    num_targets = 0

    for epoch in range(epochs):
        model.train()
        # input = train_input.copy()
        # targets = train_targets.copy()
        # training_data = zip(input, targets)
        # tqdm_train_loader = tqdm(training_data, desc="Iteration")
        tqdm_train_loader = tqdm(train_dataloader, desc="Iteration")
        for datum in tqdm_train_loader:  # input, targets
            # input = torch.tensor(datum['input'])
            # targets = torch.tensor(datum['targets'])
            input = datum['input'].to(device)
            targets = datum['targets'].to(device)
            # input = input.to(device)
            # targets = targets.to(device)

            model.zero_grad()

            scores = model(input)
            targets = targets.view(-1)
            loss = loss_fn(scores, targets)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(scores, dim=1)
            correct = torch.sum(torch.eq(preds, targets))
            train_correct += correct
            num_targets += len(targets)
            tqdm_train_loader.set_description_str(f"[Acc]: {(train_correct / num_targets):.4f}")


def evaluate(test_data, model, char_to_idx, idx_to_char):
    preds_list = []
    inputs = []
    for i, sentence in enumerate(test_data):
        inputs.append([char_to_idx[c] if c in char_to_idx else char_to_idx["<UNK>"] for c in sentence])


    model.eval()
    with torch.no_grad():
        for input in inputs:
            input = torch.tensor(input).unsqueeze(0)
            input = input.to(device)
            scores = model(input)
            prob = F.softmax(scores[-1], dim=0).data
            char_indices = torch.topk(prob, 3, dim=0)[1]
            preds_chars = ''
            for c_idx in char_indices:
                index = c_idx.item()
                if index == 1:
                    index = random.randint(2, len(char_to_idx) - 1)
                preds_chars += idx_to_char[index]
            preds_list.append(preds_chars)

    return preds_list


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'dev', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    is_cuda = torch.cuda.is_available()

    EMBEDDING_DIM = 512
    HIDDEN_DIM = 512
    BATCH_SIZE = 64

    # Check if GPU is available
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)

        print('Loading training data')
        input, targets, char_to_idx, idx_to_char = load_training_data()
        print('input targets:', input)
        # print('INPUT SHAPE:', input.shape, targets.shape)
        train_data = Text_Dataset(input, targets, 100, BATCH_SIZE)
        # print('train_data:', len(train_data))
        train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
        # train_data = zip(input, targets)
        print('Instatiating model')
        model = LSTMGenerator(EMBEDDING_DIM, HIDDEN_DIM, len(char_to_idx), len(char_to_idx)).to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.05)
        print('Training')
        # train(input, targets, model, loss_function, optimizer)
        train(train_dataloader, model, loss_function, optimizer)
        print('Saving model')
        save(model, char_to_idx, idx_to_char, args.work_dir)
    elif args.mode == "dev":
        idx_to_char, char_to_idx = load_dictionary(args.work_dir)
        model = LSTMGenerator(EMBEDDING_DIM, HIDDEN_DIM, len(char_to_idx), len(char_to_idx)).to(device)
        load(model, args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = load_test_data(args.test_data)
        print('Making predictions')
        rnn_preds = evaluate(test_data, model, char_to_idx, idx_to_char)
        print('Writing predictions to {}'.format(args.test_output))
        write_pred(rnn_preds, args.test_output)
    elif args.mode == 'test':
        print('Loading model')
        idx_to_char, char_to_idx = load_dictionary(args.work_dir)
        model = LSTMGenerator(EMBEDDING_DIM, HIDDEN_DIM, len(char_to_idx), len(char_to_idx)).to(device)
        load(model, args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = load_test_data(args.test_data)
        print('Making predictions')
        rnn_preds = evaluate(test_data, model, char_to_idx, idx_to_char)
        print('Writing predictions to {}'.format(args.test_output))
        write_pred(rnn_preds, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))

