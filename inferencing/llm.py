import cohere
import openai
import torch 
from pathlib import Path 
from typing import List 
from retry import retry 
from transformers import pipeline


class LLM:
    def __init__(self, model):
        self.model = model 
        self.max_tokens = 128

        hf_model_names = {
            "gpt2": "gpt2-large",
            "pythia-2.8B": "EleutherAI/pythia-2.8b",
            "llama2-7B": "meta-llama/Llama-2-7b-hf",
            "llama2-13B": "meta-llama/Llama-2-13b-hf"
        }
        if model == "chat":
            self.openai_client = openai.OpenAI(api_key=Path("openai_api_key.txt").read_text())
            
        elif model == "cohere":
            self.cohere_client = cohere.Client(Path("cohere_api_key.txt").read_text().strip())
            
        elif model in hf_model_names.keys():
            model_fullname = hf_model_names[model]
            device_number = 0 if torch.cuda.is_available() else -1
            self.hf_pipeline = pipeline(
                task="text-generation", 
                model=model_fullname,
                torch_dtype=torch.float16, 
                device=device_number,
                trust_remote_code=True
            )
        else:    
            raise ValueError("Model {} not supported!".format(model))

    def make_query(self, prompt):
        if self.model == "chat":
            return self.make_chatgpt_query(prompt)
        elif self.model == "cohere":
            return self.make_cohere_query(prompt)
        else:
            return self.make_huggingface_pipeline_query(prompt)
    

    @retry(tries=2, delay=1)
    def make_chatgpt_query(self, prompt):
        resp = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens
        )
        return resp.choices[0].message.content

    @retry(tries=2, delay=1)
    def make_cohere_query(self, prompt):
        resp = self.cohere_client.generate(
            prompt,
            model="command",
            max_tokens=self.max_tokens
        )
        return resp[0].text 

    def make_huggingface_pipeline_query(self, prompt):
        completion = self.hf_pipeline(
            prompt, 
            return_full_text=False,
            max_new_tokens=128, 
            num_return_sequences=1
        )
        return completion[0]["generated_text"]

if __name__ == "__main__":
    dummy_prompt = "Tell me a story"

    # llm = LLM("chat")
    # s = llm.make_query(dummy_prompt)
    # print("ChatGPT story:", s)

    # llm = LLM("cohere")
    # s = llm.make_query(dummy_prompt)
    # print("Cohere story:", s)

    # llm = LLM("gpt2")
    # s = llm.make_query(dummy_prompt)
    # print("GPT2 story:", s)

    llm = LLM("pythia-2.8B")
    s = llm.make_query(dummy_prompt)
    print("Pythia-2.8B story:", s)