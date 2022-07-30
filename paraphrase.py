import torch
import pickle 
import numpy as np
from model import Encoder, Decoder, Seq2Seq
from nltk.tokenize import word_tokenize
from dataloader import Dataloader

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

data = Dataloader("data/0_data_index_word.json", "data/0_data_prepo_new.h5") 

INPUT_DIM = data.get_vocab_size()
OUTPUT_DIM = data.get_vocab_size()
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, 
            HID_DIM, 
            ENC_LAYERS, 
            ENC_HEADS, 
            ENC_PF_DIM, 
            ENC_DROPOUT, 
            DEVICE)

dec = Decoder(OUTPUT_DIM, 
            HID_DIM, 
            DEC_LAYERS, 
            DEC_HEADS, 
            DEC_PF_DIM, 
            DEC_DROPOUT, 
            DEVICE)
model = Seq2Seq(enc, dec, 1, 1, DEVICE).to(DEVICE)

def paraphrase(sentence, model, device, max_len = 30): # len same as encode in preprocess.py
	vocab = {}

	with open("vocab.txt", "rb") as vocab_open:
		vocab = pickle.load(vocab_open)
		vocab_open.close()

	itow = {i+1:w for i,w in enumerate(vocab)} # 1-indexed vocab translation table
	wtoi = {w:i+1 for i,w in enumerate(vocab)} # Bag of words model

	if 0 not in itow:
		itow[0] = "UNK"
		
	model.eval()
		
	# sentence = sentence.lower()
	# print("lowered sentence: ", sentence)
	
	tokens = word_tokenize(sentence)

	if 0 not in itow:
		itow[0] = "UNK"

	for i,word in enumerate(tokens):
		tokens[i] = (word if word in vocab else "UNK")

	dict_len = len(itow)
	EOS, PAD, SOS = dict_len, dict_len + 1, dict_len +2
	itow[EOS] = "<EOS>"
	itow[SOS] = "<SOS>"
	itow[PAD] = "<PAD>"

	wtoi["<EOS>"] = EOS
	wtoi["<SOS>"] = SOS

	tokens = ["<SOS>"] + tokens + ["<EOS>"]
		
	#src_indexes = [src_field.vocab.stoi[token] for token in tokens]
	
	src_indexes = np.zeros(max_len, dtype = "uint32")
	for i, word in enumerate(tokens):
		if i< max_len:
			tokens[i] = wtoi[word]

	print(tokens)
	src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
	print("src_tensor shape", src_tensor.shape)
	src_mask = model.make_src_mask(src_tensor)
	
	with torch.no_grad():
		enc_src = model.encoder(src_tensor, src_mask)

	#trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
	trg_indexes = [wtoi["<SOS>"]]
	print("trg_indexes", trg_indexes)
	for i in range(max_len):
		trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
		print("trg_tensor shape",trg_tensor.shape)
		trg_mask = model.make_trg_mask(trg_tensor)
		
		with torch.no_grad():
			output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
		
		pred_token = output.argmax(2)[:,-1].item()
		
		trg_indexes.append(pred_token)

		if pred_token == wtoi["<EOS>"]:
			break
	
	trg_tokens = [itow[i] for i in trg_indexes]
	return ' '.join(trg_tokens[1: -1]), attention

if __name__ == "__main__":
    src = "The physical topology defines how nodes communicate in a network over its logical topology"
    print("Original: ", src)

    model.load_state_dict(torch.load("test_model.pt")["model"])
    translation, _ = paraphrase(src, model, DEVICE)

    print("Paraphrase:", translation)
