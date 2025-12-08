from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct", 
    task="text-generation",
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Write a five line summary on the following text. /n {text}",
    input_variables=["text"],
)


parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'Unemployment in Nepal'})

print(result)

chain.get_graph().print_ascii()