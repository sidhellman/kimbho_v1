import torch

# Load model + checkpoint
model = DecoderLanguageModel(...).to("cuda")
model.load_state_dict(torch.load("checkpoint_epoch_3.pt"))
model.eval()

prompt_ids = torch.tensor([[5, 67, 23]], dtype=torch.long).cuda()  # (batch_size=1, seq_len=3)
max_new_tokens = 20

for _ in range(max_new_tokens):
    logits = model(prompt_ids)  # shape: (1, current_seq_len, vocab_size)
    next_token_logits = logits[:, -1, :]  # only the last position
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # argmax for greedy
    prompt_ids = torch.cat([prompt_ids, next_token], dim=1)

print("Generated token sequence:", prompt_ids[0].tolist())
