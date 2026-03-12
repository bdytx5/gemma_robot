from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
print("chat_template:", tokenizer.chat_template)
print("special_tokens:", tokenizer.all_special_tokens)
print("vocab sample:", {k: v for k, v in list(tokenizer.get_vocab().items()) if 'turn' in k or 'start' in k or 'end' in k or 'bos' in k or 'eos' in k})
