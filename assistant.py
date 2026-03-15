import ollama
response = ollama.chat(model='qwen3.5:9b', messages= [
    {
        'role': 'user',
        'content': 'Why is the sky blue?',
    },
])
print(response['message']['content'])