from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    temperature=0.7,
)
model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

# More explicit template with examples
template = PromptTemplate(
    template='''Generate the name, age and city of a fictional {place} person.

Return ONLY a valid JSON object with the actual data, not a schema. 
Example format:
{{"name": "John Doe", "age": 25, "city": "New York"}}

{format_instruction}

Output:''',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

try:
    final_result = chain.invoke({'place': 'sri lankan'})
    print(final_result)
    print(f"\nName: {final_result.name}")
    print(f"Age: {final_result.age}")
    print(f"City: {final_result.city}")
except Exception as e:
    print(f"Error: {e}")