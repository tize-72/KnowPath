import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
import logging
import ollama

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

RETRY_EXCEPTIONS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)

def retry_on_exception(func):
    return retry(
        stop=stop_after_attempt(5),  
        wait=wait_exponential(multiplier=1, min=4, max=10),  
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
        reraise=True
    )(func)


def clean_text(text: str) -> str:
    """Clean non-ASCII characters from text"""
    # Replace common special quotes and punctuation
    replacements = {
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '–': '-',
        '—': '-',
        '…': '...',
        '•': '*',
        '°': ' degrees ',
        '×': 'x',
        '÷': '/',
        '≠': '!=',
        '≤': '<=',
        '≥': '>=',
        '±': '+/-',
        '∞': 'infinity',
        '′': "'",
        '″': '"',
        '€': 'EUR',
        '£': 'GBP',
        '¥': 'JPY',
        '©': '(c)',
        '®': '(R)',
        '™': '(TM)',
    }
    
    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove remaining non-ASCII characters
    cleaned_text = ''.join(char if ord(char) < 128 else ' ' for char in text)
    
    # Clean extra spaces
    cleaned_text = ' '.join(cleaned_text.split())
    
    return cleaned_text
@retry_on_exception
def custom_embedding(texts,api_key):
    # Clean and check input text
    cleaned_texts = []
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            text = str(text)
        # Clean text
        cleaned_text = clean_text(text)
        cleaned_texts.append(cleaned_text)
    batch_size = 20
    from openai import OpenAI
    API_SECRET_KEY = "your api key";
    BASE_URL = "http"
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    # client = OpenAI(api_key)
    
    texts = [str(text).strip() for text in cleaned_texts]
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        try:
            response = client.embeddings.create(
                model="text-embedding-v4",
                input=batch_texts,
                encoding_format="float",
                dimensions=1536,
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
        except Exception as e:
            print(f"处理批次 {i} 时出错: {str(e)}")
            empty_embedding = [0.0] * 1536
            embeddings.extend([empty_embedding] * len(batch_texts))
    return embeddings

@retry_on_exception
def custom_llm(prompt, api_key):
    client = OpenAI(api_key)
    if isinstance(prompt, dict):
        formatted_prompt = prompt.get("text", "")
    elif isinstance(prompt, str):
        formatted_prompt = prompt
    else:
        formatted_prompt = str(prompt)
    completion = client.chat.completions.create(
    model="gpt-4o-mini", # 
    messages=[
        {'role': 'user', 'content': formatted_prompt}],
    temperature=0.2,
    )
    return completion.choices[0].message.content


def run_ollama(prompt, openai_api_keys='', llm="qwen2.5:7b", max_tokens=512, engine="qwen2.5:7b"):
    """ollama方式运行大模型

    Args:
        prompt (_type_): 提示词
        temperature (_type_): 温度系数 0.8默认 可以更高， 使得回复更具有创意
        max_tokens (_type_): 大模型回复的最大token数量
        openai_api_keys (str, optional): openai api字符串. Defaults to ''.
        engine (str, optional): 模型类型. Defaults to "qwen2".

    Returns:
        _type_: 返回大模型推理结果
    """
    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)

    if 'gpt' in engine:
        if isinstance(prompt, dict):
            formatted_prompt = prompt.get("text", "")
        elif isinstance(prompt, str):
            formatted_prompt = prompt
        else:
            formatted_prompt = str(prompt)
        messages=[
        {'role': 'user', 'content': formatted_prompt}]
        from openai import OpenAI
        client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
        completion = client.chat.completions.create(
        # model="gpt-3.5-turbo",# gpt-4o-mini
        model=engine,# gpt-4o-mini
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
        )
        print(completion.choices[0].message.content)
        result = completion.choices[0].message.content
        token_num = {"total": completion.usage.total_tokens, 
                     "input": completion.usage.prompt_tokens, 
                     "output": completion.usage.completion_tokens}
    elif "deep" in engine:
        API_SECRET_KEY = "sk-e31287dd7a6848708ebcf6c407b9de6d";
        BASE_URL = "https://api.deepseek.com"
        client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
        if isinstance(prompt, dict):
            formatted_prompt = prompt.get("text", "")
        elif isinstance(prompt, str):
            formatted_prompt = prompt
        else:
            formatted_prompt = str(prompt)
        messages=[
        {'role': 'user', 'content': formatted_prompt}]
        from openai import OpenAI
        completion = client.chat.completions.create(
        model="deepseek-chat",# gpt-4o-mini
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
        )
        print(completion.choices[0].message.content)
        result = completion.choices[0].message.content
        token_num = {"total": completion.usage.total_tokens, 
                     "input": completion.usage.prompt_tokens, 
                     "output": completion.usage.completion_tokens}
    else:
        if isinstance(prompt, dict):
            formatted_prompt = prompt.get("text", "")
        elif isinstance(prompt, str):
            formatted_prompt = prompt
        else:
            formatted_prompt = str(prompt)
        messages=[
        {'role': 'user', 'content': formatted_prompt}]
        response = ollama.chat(model=engine, 
                                messages=messages, 
                                options={
                                "temperature":0.2, # default 0.8 模型的温度。增加温度将使模型的回答更具创意
                                "top_k" : 40, # default 40 降低产生无意义答案的概率。值越高，答案多样化越强
                                "top_p" : 0.9, # default 0.9 与 top-k 配合使用。控制模型回答的自由度，值越高，自由度越高
                                "num_predict" : max_tokens, #default 128 生成文本时要预测的最大标记数。最大的预测tokens数
                                "num_ctx" : 2048 # 设置用于生成下一个标记的上下文窗口的大小。（默认值：2048）
                                },
                                keep_alive = '10m', # 让模型在内存中存在一分钟
                                )
        print(response['message']['content'])
        token_num = {"total": response['prompt_eval_count']+response['eval_count'], 
                 "input": response['prompt_eval_count'], 
                 "output": response['eval_count']}
        result = response['message']['content']

    return result

class CustomEmbeddings:
    def __init__(self, api_key):
        self.api_key = api_key

    def embed_documents(self, texts):
        return custom_embedding(texts, self.api_key)

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def __call__(self, text):
        if isinstance(text, str):
            return self.embed_query(text)
        elif isinstance(text, list):
            return self.embed_documents(text)
        else:
            raise ValueError("Input must be a string or a list of strings") 