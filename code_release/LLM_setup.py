import openai
import os
import copy
import json
import requests

T = 0.0
OPENAI_API_KEY = "set your openai key here"
BARD_TOKEN='set your bard token here'
palm_token = 'set your palm token here'
claude_api_key = "set your claude api key here"

'''
OSS LLMs:

- openchat-13b, wizardlm-13b, vicuna-13b, vicuna-7b, oasst-12b, chatglm2-6b
1. Install [fastchat](https://github.com/lm-sys/FastChat) and download the OSS LLM models from their official websites.  
2. Use fastchat to start an api server (see [here](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md)).  
3. Then you can use the following functions to call the LLMs.

- llama2-13b, llama2-7b:
1. Install [llama2-flask-api](https://github.com/unconv/llama2-flask-api) and download the Llama models from their official websites.  
2. Start an api server (see [here](https://github.com/unconv/llama2-flask-api)).  
3. Then you can use the following functions to call the LLMs.

API LLMs:

- Chat models (gpt-3.5-turbo-1106, gpt-4-1106-preview, gpt-4o)
1. set the openai key above
2. Then you can use the following functions to call the LLMs.

- Completion models (bard, text-bison-001, text-davinci-003, text-davinci-002, claude-instant)
1. set the api key above
2. Then you can use the following functions to call the LLMs.
'''


def get_completion(prompt, model, temperature=0.0):
    if model in ["openchat-13b", "wizardlm-13b", "vicuna-13b", "vicuna-7b", "oasst-12b", "chatglm2-6b"]:
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message["content"]
    
    elif model in ["llama2-13b", "llama2-7b"]:
        messages = [{"role": "user", "content": prompt}]
        response = requests.post("http://localhost:5000/chat", json={"messages":messages})
        response = json.loads(response.text)
        return response['choices'][0]['message']["content"]


    elif model in ["gpt-3.5-turbo-1106", "gpt-4-1106-preview", "gpt-4o"]:
        openai.api_key = OPENAI_API_KEY
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message["content"]
    
    elif model in ["bard", "text-bison-001", "text-davinci-003", "text-davinci-002", "claude-instant"]:
        from bardapi import Bard
        import google.generativeai as palm
        import asyncio
        from fastapi_poe.types import ProtocolMessage
        from fastapi_poe.client import get_bot_response
        message_prompt = prompt
        

        if model == 'text-davinci-003' or model == 'text-davinci-002':
            response = openai.Completion.create(
                model=model,
                prompt = message_prompt,
                temperature=0,
                max_tokens=1000
            )
            out=response.choices[0].text.strip() 

        elif model == 'text-bison-001': 
            os.environ['HTTP_PROXY'] = 'http://127.0.0.1:33210'
            try:
                palm.configure(api_key=palm_token)
                completion = palm.generate_text(
                    model='models/text-bison-001',
                    prompt=message_prompt,
                    temperature=0,
                    max_output_tokens=1000,
                )
                out=completion.result
            except Exception as e:
                print(f"{model} API failed with error: {str(e)}.")
            finally:
                # Restore the original value of HTTP_PROXY
                os.environ.clear()  # Clear all environment variables
                os.environ.update(os.environ.copy())

        elif model == 'bard':
            try:
                # bard_token = input("Enter bard token (__Secure-1PSID): ")
                bard_token = BARD_TOKEN
                bard = Bard(token=bard_token)
                out=bard.get_answer(message_prompt)['content']
                return out
            except Exception as e:
                print(f"Bard API failed with error: {str(e)}. Please wait a while and try again.")
        
        elif model=='Claude-Instant-Temp0':
            def get_claude_responses(api_key, model, message_prompt):
                complete_response = ''
                message = ProtocolMessage(role="user", content=message_prompt)
                for partial in get_bot_response(messages=[message], 
                                                    bot_name=model, 
                                                    api_key=api_key):

                    complete_response+=partial.text
                return complete_response
            try:
                api_key = claude_api_key
                out=get_claude_responses(api_key, model, message_prompt)
                return out
            except Exception as e:
                print(f"{model} API failed with error: {str(e)}.")
                


class ChatApp:
    def __init__(self, system_message, init_message=None):
        self.messages = [
            {"role": "system", "content": system_message},
        ]

    def chat(self, message, model,temperature=0):
        self.messages.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
                    model=model,
                    messages=self.messages,
                    temperature=temperature,
                )            
        self.messages.append({"role": "assistant", "content": response["choices"][0]["message"].content})
        return response["choices"][0]["message"]['content']
    
    def chat_wo_update(self, message, model, temperature=0):
        tmp_message=copy.deepcopy(self.messages)
        tmp_message.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
                    model=model,
                    messages=tmp_message,
                    temperature=temperature,
                )

        return response["choices"][0]["message"]['content']
