import openai
openai.api_key = "YOUR_API_KEY_HERE"

def is_pornographic(url):
    prompt = f"Is {url} a pornographic website?"
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.5,
        max_tokens=1024,
        n=1,
        stop=None,
        timeout=60,
    )
    answer = response.choices[0].text.strip().lower()
    return answer == "yes"