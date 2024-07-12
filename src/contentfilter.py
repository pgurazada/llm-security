import os

from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict


def run_moderation(input: str) -> Dict:

    load_dotenv()

    client = OpenAI(
        api_key=os.environ['OPENAI_KEY']
    )

    response = client.moderations.create(
        input=input
    )
    output = response.results[0]
    filter_results = output.categories.model_dump()

    return filter_results

if __name__ == '__main__':

    filter_results = run_moderation("This is a sample input")

    for key in filter_results.keys():
        print(f"Moderation for {key} is {filter_results[key]}")