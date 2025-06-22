from llmlingua import PromptCompressor
from transformers import AutoTokenizer, AutoModelForTokenClassification

# compressor = PromptCompressor(
#     model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
#     use_llmlingua2=True
# )
#
# original_prompt = """John: So, um, I've been thinking about the project, you know, and I believe we need to, uh, make some changes. I mean, we want the project to succeed, right? So, like, I think we should consider maybe revising the timeline.
# Sarah: I totally agree, John. I mean, we have to be realistic, you know. The timeline is, like, too tight. You know what I mean? We should definitely extend it.
# """
# results = compressor.compress_prompt_llmlingua2(
#     original_prompt,
#     rate=0.6,
#     force_tokens=['\n', '.', '!', '?', ','],
#     chunk_end_tokens=['.', '\n'],
#     return_word_label=True,
#     drop_consecutive=True
# )
#
# print(results.keys())
# print(f"Compressed prompt: {results['compressed_prompt']}")
# print(f"Original tokens: {results['origin_tokens']}")
# print(f"Compressed tokens: {results['compressed_tokens']}")
# print(f"Compression rate: {results['rate']}")
#
# # get the annotated results over the original prompt
# word_sep = "\t\t|\t\t"
# label_sep = " "
# lines = results["fn_labeled_original_prompt"].split(word_sep)
# annotated_results = []
# for line in lines:
#     word, label = line.split(label_sep)
#     annotated_results.append((word, '+') if label == '1' else (word, '-')) # list of tuples: (word, label)
# print("Annotated results:")
# for word, label in annotated_results[:10]:
#     print(f"{word} {label}")

tokenizer = AutoTokenizer.from_pretrained("microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")
max_length = tokenizer.model_max_length
print(f"最大输入长度: {max_length} tokens")