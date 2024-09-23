from factscore.lm import LM
import google.generativeai as genai
import sys
import time
import os
import numpy as np
import logging

class GoogleGeminiModel(LM):
    def __init__(self, model_name, cache_file=None, key_path="api.keys", temperature=0.7):
        self.model_name = model_name
        self.client = None
        self.api_key = None
        self.key_path = key_path
        self.temperature = temperature
        self.save_interval = 100
        super().__init__(cache_file)

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        cache_file = kwargs.get("cache_file", None)
        key_path = kwargs.get("key_path", "api.key")
        instance = cls(model_name=model_name_or_path, cache_file=cache_file, key_path=key_path)
        return instance

    def load_model(self):
        # load api key
        key_path = self.key_path
        assert os.path.exists(key_path), f"Please place your Gemini APT Key in {key_path}."
        if self.api_key is None:
            with open(key_path, 'r') as f:
                keys = dict(line.strip().split('=') for line in f if line.strip())
            # Google Gemini API 키 설정
            self.api_key = keys.get('gemini', None)
            if not self.api_key:
                raise ValueError("API key is required to use Google Gemini API.")
        
        self.model = self.model_name 
        genai.configure(api_key=self.api_key.strip())
        self.client = genai.GenerativeModel(self.model_name)

        print(f'... Model loading... {self.model_name}')

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        if self.add_n % self.save_interval == 0:
            self.save_cache()

        # time.sleep(4) # 분당 허용치를 넘지 않기 위한 처리 (err. message: google.api_core.exceptions.ResourceExhausted: 429 Resource has been exhausted (e.g. check quota). )
        
        # https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=python&authuser=1&hl=ko 참고
        # safety_settints 설정
        safe = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        response = self.client.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_output_length,
                        temperature=self.temperature),
                    safety_settings=safe
                    )
        
        output = response.text
        return output, response
