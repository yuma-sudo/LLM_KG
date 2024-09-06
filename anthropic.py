import anthropic
from pydantic import BaseModel
import numpy as np
import pickle

# Initialize the Anthropic client with your API key
client = anthropic.Anthropic(api_key="your-pass-key")

# Define the ProductRelationshipExplanation class to parse the response
class ProductRelationshipExplanation(BaseModel):
    relationship: str
    explanation: str

# Define the ScenarioSimulation class to parse the scenario generation response
class ScenarioSimulation(BaseModel):
    scenarios: list[str]

# Function to simulate scenarios using the Anthropic API
def simulate_scenarios(first_type):
    # Define the system message
    system_message = "You are a creative assistant tasked with generating shopping scenarios."
    
    # Construct the conversation using the Messages API structure
    messages = [
        {"role": "user", "content": f"Generate a shopping scenario where the customer is looking for products on "
                                    f"the e-commerce website eBay in the {first_type} category. Make sure the scenario is clear and provides "
                                    "context on why the customer is shopping for these items. Try to make the scenarios diverse but also "
                                    "realistic. Generate five different shopping scenarios involving {first_type}."}
    ]
    
    # Call the Messages API
    response = client.messages.create(
        model="claude-3-haiku-20240307",  # Specify the model to use
        max_tokens=1000,  # Set the maximum number of tokens in the response
        messages=messages,  # Pass the conversation history
        system=system_message  # Provide the system message as a separate parameter
    )
    
    # Print the content of the response
    scenarios = response.content[0].text.split('\n\n')[1:]
    return scenarios

# Function to get the relationship and explanation using the Anthropic API
def get_relationship(scenario, first_type, second_type):
    response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1000,
    temperature=0,
    system="You are a product relationship extractor.",
    messages=[
        {"role": "user", "content": f"The scenario is as follows: {scenario}. In this case, what is "
                                    f"the relationship of {first_type} and the second category {second_type}? The relationship can "
                                    f"be 'substitute' (are similar and could replace each other), 'complementary' (frequently go "
                                    f"together as complements to each other), or 'irrelevant' (irrelevant to each other)."},
        ]
    )
    
    # Extract the first sentence from the response as the relationship
    if('irrelevant' in response.content[0].text.split('.')[0]):
        relationship_learned = 'irrelevant'
    elif('complementary' in response.content[0].text.split('.')[0]):
        relationship_learned = 'complementary'
    elif('substitute' in response.content[0].text.split('.')[0]):
        relationship_learned = 'substitute'
        
    return relationship_learned