import openai
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

# Example: last 30 days
start_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
end_date = datetime.utcnow().strftime("%Y-%m-%d")

usage = openai.Image.list(
    start_date=start_date,
    end_date=end_date
)

print(usage)