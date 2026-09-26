from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("models/qwen2.5-0.5b-instruct")

text = "The qucik brown fox jumps over the lazy dog"
token_ids = tokenizer.encode(text)
print("token IDs:", token_ids)

decoded = tokenizer.decode(token_ids)
print("decoded:", decoded)

assert decoded.strip() == text.strip(), "Round-trip mismatch!"
print("tokenizer round-trip successful")
print("BOS:", tokenizer.bos_token, tokenizer.bos_token_id)
print("EOS:", tokenizer.eos_token, tokenizer.eos_token_id)
print("Pad:", tokenizer.pad_token, tokenizer.pad_token_id)
