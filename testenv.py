from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY:", openai_api_key)