import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

def generate_response(prompt, model_engine="davinci", max_tokens=60, temperature=0.7, n=1, stop=None):
    try:
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
            stop=stop
        )
        ai_response = response.choices[0].text.strip()
        return ai_response
    except Exception as e:
        print(f"Error generating response: {e}")
        return None
