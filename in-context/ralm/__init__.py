# from transformers import GPT2Tokenizer, GPT2Model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
# model = GPT2Model.from_pretrained('gpt2-large')
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)

"""---------------------------------------------------------------------------------------------------------------------------------------------------------"""
# from transformers import GPT2Tokenizer, GPT2Model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
# model = GPT2Model.from_pretrained('gpt2-xl')
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)

"""---------------------------------------------------------------------------------------------------------------------------------------------------------"""
# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

"""---------------------------------------------------------------------------------------------------------------------------------------------------------"""
# Load model directly   deepseek-ai/deepseek-llm-7b-chat
# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")

"""---------------------------------------------------------------------------------------------------------------------------------------------------------"""
# from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
#
# tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")
# model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
# input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]
# embeddings = model(input_ids).pooler_output

"""---------------------------------------------------------------------------------------------------------------------------------------------------------"""
# import torch
# from transformers import AutoTokenizer, AutoModel
#
# tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
# model = AutoModel.from_pretrained('facebook/contriever')
#
# sentences = [
#     "Where was Marie Curie born?",
#     "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
#     "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
# ]

# # Apply tokenizer
# inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
#
# # Compute token embeddings
# outputs = model(**inputs)
#
# # Mean pooling
# def mean_pooling(token_embeddings, mask):
#     token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
#     sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
#     return sentence_embeddings
# embeddings = mean_pooling(outputs[0], inputs['attention_mask'])

"""---------------------------------------------------------------------------------------------------------------------------------------------------------"""
from transformers import GPT2Tokenizer

# 加载 GPT-2 分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 文本
text = '''

Wilhelm Röntgen received the first Nobel Prize in Physics in recognition of his extraordinary services. It is one of the five Nobel Prizes established by Alfred Nobel in 1895 and awarded since 1901.

'''

# 计算 token 数
tokens = tokenizer.encode(text)
print(len(tokens))
