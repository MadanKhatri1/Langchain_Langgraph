from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableBranch, RunnableLambda
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    temperature=0.1,  
    max_new_tokens=100,
)
model = ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field(
        description="The sentiment of the feedback"
    )

json_parser = PydanticOutputParser(pydantic_object=Feedback)
text_parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="""You are a sentiment classifier. Analyze the feedback and respond ONLY with valid JSON.

Feedback: {feedback}

Respond with JSON in this exact format:
{{"sentiment": "Positive"}} or {{"sentiment": "Negative"}}

Do not include any explanation, just the JSON object.

JSON Response:""",
    input_variables=["feedback"]
)

prompt2 = PromptTemplate(
    template="""Write a brief, professional response to this positive feedback (maximum 1 sentences):

Feedback: {feedback}

Response:""",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="""Write a brief, professional and empathetic response to this negative feedback (maximum 1 sentences):

Feedback: {feedback}

Response:""",
    input_variables=["feedback"]
)

classifier_chain = prompt1 | model | json_parser

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'Positive', prompt2 | model | text_parser),
    (lambda x: x.sentiment == 'Negative', prompt3 | model | text_parser),
    RunnableLambda(lambda x: "Could not classify the sentiment.")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'The product quality is outstanding and exceeded my expectations!'}))