

CHAT_TEMPLATE = """
    {% for message in messages %}
        {% if message['role'] == 'system' %}
            <|im_start|>system\n{{ message['content'].strip() }}\n<|im_end|>\n
        {% elif message['role'] == 'user' %}
            <|im_start|>user\n{{ message['content'].strip() }}\n<|im_end|>\n
        {% elif message['role'] == 'assistant' %}
            {% generation %}<|im_start|>assistant\n{{ message['content'].strip() }}\n<|im_end|>{% endgeneration %}
        {% endif %}
    {% endfor %}
"""

    # {%- if add_generation_prompt %}
    #     {{- '<|assistant|>\n' }}
    # {%- endif %}
    
"""
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- '<|im_start|>user\n' + message['content'].strip() + '\n<|im_end|>\n' }}
    {%- elif message['role'] == 'system' %}
        {{- '<|im_start|>system\n' + message['content'].strip()  + '\n<|im_end|>\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '<|im_start|>assistant\n'  + message['content'] + '\n<|im_end|>' + eos_token }}
    {%- endif %}
{%- endfor %}
"""