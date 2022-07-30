import h5py
import json
import torch
import torch.utils.data as data
 
class Dataloader(data.Dataset):
    def __init__(self, input_json_file_path, input_h5_path):
        super(Dataloader, self).__init__()
        #print("Reading", input_json_file_path)

        with open(input_json_file_path) as input_file:
            data_dict = json.load(input_file)

        self.ix_to_word = {}

        for k in data_dict["index_to_word"]:
            self.ix_to_word[int(k)] = data_dict["index_to_word"][k]

        self.UNK_token = 0

        if 0 not in self.ix_to_word:
            self.ix_to_word[0] = "<UNK>"

        else :
            print("0 in self.ix to word")
            raise Exception

        self.EOS_token = len(self.ix_to_word)
        self.ix_to_word[self.EOS_token] = "<EOS>"
        self.PAD_token = len(self.ix_to_word)
        self.ix_to_word[self.PAD_token] = "<PAD>"
        self.SOS_token = len(self.ix_to_word)
        self.ix_to_word[self.SOS_token] = "<SOS>"
        self.vocab_size = len(self.ix_to_word)
        
        print("DataLoader loading h5 question file:", input_h5_path)
        # TRAINING SET... 
        qa_data = h5py.File(input_h5_path, "r")
        # question ids
        src, src_len = self.pad_data(torch.from_numpy(qa_data["sentences_array"][...].astype(int)), torch.from_numpy(qa_data["sentences_length"][...].astype(int)))
        trg, trg_len = self.pad_data(torch.from_numpy(qa_data["paraphrases_array"][...].astype(int)), torch.from_numpy(qa_data["paraphrases_length"][...].astype(int)))
        print("Finish proceesss data... :)")
        self.train_id = 0
        self.seq_length = src.size()[1]

        print("Training dataset length : ", src.size()[0])

        self.test_id = 0

        qa_data.close()

        self.src = src
        self.len = src_len
        self.trg = trg
        self.trg_len = trg_len

        print('len self.src', len(self.src))
        print('len self.len', len(self.len))
        print('len self.trg', len(self.trg))
        print('len self.trg_len', len(self.trg_len))

    def pad_data(self, data, data_len):
        # https://stackoverflow.com/questions/48686945/reshaping-a-tensor-with-padding-in-pytorch
        print("Padding data..")
        N = data.size()[0]
        print("N: ", N)
        new_data = torch.zeros(N, data.size()[1] + 2, dtype=torch.long) + self.PAD_token
        for i in range(N):
            new_data[i, 1:data_len[i]+1] = data[i, :data_len[i]]
            new_data[i, 0] = self.SOS_token
            new_data[i, data_len[i]+1] = self.EOS_token
            data_len[i] += 2
        return new_data, data_len

    def __len__(self):
        return self.len.size()[0]

    def __getitem__(self, idx):
        return (self.src[idx], self.len[idx], self.trg[idx], self.trg_len[idx])

    def get_vocab_size(self):
        return self.vocab_size

    def get_seq_length(self):
        return self.seq_length