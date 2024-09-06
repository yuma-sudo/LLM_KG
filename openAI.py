from pydantic import BaseModel
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import networkx as nx
import numpy as np
import pickle
import random
import os


os.environ['OPENAI_API_KEY'] = 'your-pass-key'
client = OpenAI()

# Define the ProductRelationshipExplanation class to parse the response
class ProductRelationshipExplanation(BaseModel):
    relationship: str
    explanation: str

# Define the ScenarioSimulation class to parse the scenario generation response
class ScenarioSimulation(BaseModel):
    scenarios: list[str]

# Function to simulate scenarios using the OpenAI API
def simulate_scenarios(first_type):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": f"You are a creative assistant tasked with generating \
            shopping scenarios. Generate a shopping scenario where the customer is looking for \
            products on the e-commerce website eBay in the {first_type} category. Make sure the \
            scenario is clear and provides context on why the customer is shopping for these \
            items. Try to make the scenarios diverse but also realistic. "},
            {"role": "user", "content": f"Generate five different shopping scenarios involving {first_type}."}
        ],
        response_format=ScenarioSimulation,
    )
    scenarios = completion.choices[0].message.parsed.scenarios
    return scenarios

# Function to get the relationship and explanation using the OpenAI API
def get_relationship(scenario, first_type, second_type):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a product relationship extractor and your decision \
            is based on the specific scenario you have encountered. Extract the relationship between \
            the two product categories. The relationship can be 'substitute' (are similar and could \
            replace each other), 'complementary' (frequently go together as complements to each other),\
            or 'irrelevant' (irrelevant to each other)."},
            # Also, provide an explanation for why you \
            # believe this is the case.
            # "},                           
            {"role": "user", "content": f"The scenario is as follows: {scenario}. In this case, \
            what is the relationship of {first_type} and the second category {second_type}"}
        ],
        response_format=ProductRelationshipExplanation,
    )
    relationship_learned = completion.choices[0].message.parsed.relationship
    return relationship_learned