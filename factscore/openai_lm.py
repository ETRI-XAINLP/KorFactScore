from factscore.lm import LM
import openai
from openai import OpenAI       # kmh
import sys
import time
import os
import numpy as np
import logging

class OpenAIModel(LM):

    def __init__(self, model_name, cache_file=None, key_path="api.key", temperature=0.7):
        self.client = None                          # kmh
        self.model_name = model_name
        self.key_path = key_path
        self.temp = temperature
        self.save_interval = 100
        super().__init__(cache_file)

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        key_path = kwargs.get("key_path", "api.key")
        cache_file = kwargs.get("cache_file", None)
        instance = cls(model_name=model_name_or_path, cache_file=cache_file, key_path=key_path)
        return instance
    
    def load_model(self):
        # load api key
        key_path = self.key_path
        assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."
        with open(key_path, 'r') as f:
            keys = dict(line.strip().split('=') for line in f if line.strip())
            # api_key = f.readline()
        # OpenAI API 키 설정
        api_key = keys.get('openai', None)
        openai.api_key = api_key.strip()

        self.model = self.model_name            # TBD
        self.client = OpenAI(api_key=api_key.strip())  # kmh

        print(f'... Model loading... {self.model_name}')

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        # return a tuple of string (generated text) and metadata (any format)
        # This should be about generating a response from the prompt, no matter what the application is
        response = call_chat_completions_v240423(prompt, self.client, model=self.model_name, temp=self.temp,
                                                 max_tokens=max_output_length)
        output = response.choices[0].message.content
        return output, response


def call_chat_completions_v240423(prompt, client, model="gpt-4", max_tokens=1024, temp=1.0, verbose=False):
    response = None
    received = False
    num_rate_errors = 0

    while not received:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "You are ChatGPT, a helpful and knowledgeable assistant."},
                          {"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temp            # kmh
            )

            received = True
        except:
            num_rate_errors += 1
            error = sys.exc_info()[0]
            # if error == openai.error.InvalidRequestError:         # kmh@240604
            #     # something is wrong: e.g. prompt too long
            #     logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
            #     assert False
            # elif error == openai.APIError:
            #     logging.critical(f"APIError\nPrompt passed in:\n\n{prompt}\n\n")
            #     assert False

            logging.error("Unexpected API error: %s (%d). Waiting %dsec" % (error, num_rate_errors, np.power(2, num_rate_errors)))
            time.sleep(np.power(2, num_rate_errors))

    #
    return response

