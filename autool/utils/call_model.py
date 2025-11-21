from openai import OpenAI
from ..config import Config

def call_model(messages, temperature=None, top_p=None, seed=None):
    client = OpenAI(
        base_url=Config.OPENAI_BASE_URL,
        api_key=Config.OPENAI_API_KEY,
    )
    
    completion = client.chat.completions.create(
        model=Config.MODEL_NAME,
        messages=messages,
        temperature=temperature or Config.TEMPERATURE,
        top_p=top_p or Config.TOP_P,
        frequency_penalty=Config.FREQUENCY_PENALTY,
        presence_penalty=Config.PRESENCE_PENALTY,
        seed=seed or Config.SEED
    )

    content = completion.choices[0].message.content
    prompt_tokens = completion.usage.prompt_tokens if completion.usage else 0
    completion_tokens = completion.usage.completion_tokens if completion.usage else 0
    print(f"debug info: prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}")
    return content, prompt_tokens, completion_tokens
    

if __name__ == "__main__":
    # --- test call_model ---
    messages = [
        {
            "role": "user",
            "content": "Good morning, how are you?"
        }
    ]
    content, prompt_tokens, completion_tokens = call_model(messages)
    print("Model Output: ", content)
    print("Prompt Tokens: ", prompt_tokens)
    print("Completion Tokens: ", completion_tokens)