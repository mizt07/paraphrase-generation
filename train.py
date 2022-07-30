from dataloader import Dataloader
from model import Encoder, Decoder, Seq2Seq
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch
from tqdm import tqdm

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

data = Dataloader("data/0_data_index_word.json", "data/0_data_prepo_new.h5") 
print("Dataset loaded. ")

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
MODEL_NAME = "test_model.pt"

def save_model(model, model_optim, epoch, save_file):
	checkpoint = {
		"epoch": epoch,
		"model": model.state_dict(),
		"model_opt": model_optim.state_dict()
	}

	torch.save(checkpoint, save_file)

def init_weights(model):
	for name, param in model.named_parameters():
		nn.init.uniform_(param.data, -0.08, 0.08)

if __name__ == "__main__":
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

	print("Model successfully loaded.")

	train_loader = Data.DataLoader(
		data,
		batch_size=32,
		shuffle=True
	)

	model.apply(init_weights)
	optimizer = optim.Adam(model.parameters(), lr = 0.0005)

	model.train()
	model.to(DEVICE)

	cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=data.PAD_token)

	print("============ START TRAINING ============")
	epochs = 15
	for epoch in range(1,epochs+1):
		epoch_loss = 0
		itr = 0
		model.train()

		for phrase, phrase_len, paraphrase, paraphrase_len in tqdm(
				train_loader, ascii=True, desc="train" + str(epoch)):
				
			# print("src", phrase.shape)
			# print("trg", paraphrase.shape)

			phrase = phrase.to(DEVICE)
			paraphrase = paraphrase.to(DEVICE)

			output, _ = model(phrase,paraphrase[:,:-1])

			#loss_1 = cross_entropy_loss(out.permute(1, 2, 0), paraphrase)

			output_dim = output.shape[-1]
			output = output.contiguous().view(-1, output_dim)
			paraphrase_i = paraphrase[:,1:].contiguous().view(-1)
			
			loss = cross_entropy_loss(output, paraphrase_i)
			optimizer.zero_grad()
			loss.backward()

			optimizer.step()

			epoch_loss += loss.item()

			itr += 1
			torch.cuda.empty_cache()

		print("Train Loss:  ", epoch_loss/ len(train_loader.dataset))

	# Save model
	print("Saving model...")
	save_model(model, optimizer, epoch, MODEL_NAME)

	print("Training Done")

