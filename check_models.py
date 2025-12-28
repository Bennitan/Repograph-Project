import google.generativeai as genai

# PASTE YOUR KEY HERE
GOOGLE_API_KEY = "AIzaSyBFssu0wBm5-8zJ0wq5S-_lMOOC2rIy1cw"

genai.configure(api_key=GOOGLE_API_KEY)

print("Checking available models for your key...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error: {e}")