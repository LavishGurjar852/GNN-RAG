# from transformers import pipeline, AutoTokenizer, LlamaTokenizer
# import torch
# from .base_language_model import BaseLanguageModel
#
# class Llama(BaseLanguageModel):
#     DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
#
#     @staticmethod
#     def add_args(parser):
#         parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path", default='meta-llama/Llama-2-7b-chat-hf')
#         parser.add_argument('--max_new_tokens', type=int, help="max length", default=512)
#         parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')
#
#     def __init__(self, args):
#         self.args = args
#         self.maximun_token = 4096 - 100
#
#     def load_model(self, **kwargs):
#         model = LlamaTokenizer.from_pretrained(self.args.model_path, use_fast=False, token="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")
#         return model
#
#     def tokenize(self, text):
#         return len(self.tokenizer.tokenize(text))
#
#     def prepare_for_inference(self, **model_kwargs):
#         self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, use_fast=False, token="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")
#         print("model: ", self.args.model_path)
#
#         model_kwargs.update({
#             'offload_folder': './offload_folder'  # Specify the offload folder
#         })
#
#         self.generator = pipeline(
#             "text-generation",
#             model=self.args.model_path,
#             tokenizer=self.tokenizer,
#             device_map="auto",
#             model_kwargs=model_kwargs,
#             torch_dtype=self.DTYPE.get(self.args.dtype, None),
#             token="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"  # Use the token argument instead of use_auth_token
#         )
#
#     @torch.inference_mode()
#     def generate_sentence(self, llm_input):
#         outputs = self.generator(llm_input, return_full_text=False, max_new_tokens=self.args.max_new_tokens)
#         return outputs[0]['generated_text']  # type: ignore


from transformers import pipeline, AutoTokenizer, LlamaTokenizer
import torch
from .base_language_model import BaseLanguageModel

class Llama(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    @staticmethod
    def add_args(parser):
        parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path", default='meta-llama/Llama-2-7b-chat-hf')
        parser.add_argument('--max_new_tokens', type=int, help="max length", default=512)
        parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')

    def __init__(self, args):
        self.args = args
        self.maximun_token = 4096 - 100

    def load_model(self, **kwargs):
        model = LlamaTokenizer.from_pretrained(self.args.model_path, use_fast=False, token="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")
        return model

    def tokenize(self, text):
        return len(self.tokenizer.tokenize(text))

    def prepare_for_inference(self, **model_kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, use_fast=False, token="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")
        print("model: ", self.args.model_path)

        # Ensure that the model is loaded entirely onto the GPU
        self.generator = pipeline(
            "text-generation",
            model=self.args.model_path,
            tokenizer=self.tokenizer,
            device=0,  # Ensure the model is loaded onto the GPU (device 0)
            model_kwargs=model_kwargs,
            torch_dtype=self.DTYPE.get(self.args.dtype, None),
            token="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"  # Use the token argument instead of use_auth_token
        )

    @torch.inference_mode()
    def generate_sentence(self, llm_input):
        outputs = self.generator(llm_input, return_full_text=False, max_new_tokens=self.args.max_new_tokens)
        return outputs[0]['generated_text']  # type: ignore
