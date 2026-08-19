import requests
import os
from dotenv import load_dotenv
load_dotenv()

response = requests.get(
    "https://artificialanalysis.ai/api/v2/language/models/free",
    headers={"x-api-key": os.environ['ARTIFICAL_ANALYSIS_API_KEY']},
    params={"page" : 1}
)
data = response.json()

for d in data['data']:
    print(f"{d['name']}: {d['evaluations']['artificial_analysis_intelligence_index']}")