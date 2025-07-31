from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser,PydanticOutputParser
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

chat_model = ChatOpenAI()

## String Output Parser
template = PromptTemplate(
    template='write detailed report on {topic}',
    input_variables=['topic']
)
template1 = PromptTemplate(
    template='write 5 line summary on the following text.\n {text}',
    input_variables=['text']
)
parser = StrOutputParser()

chain = template | chat_model | parser | template1 | chat_model | parser

chain.invoke({'topic':'black hole'})

## Json OutPut Parser
parser = JsonOutputParser()
template = PromptTemplate(
    template = 'give me name, age and city of a fictional person.\n {format_instruction}',
    input_variables = [],
    partial_variables = {'format_instruction':parser.get_format_instructions()}
)
chain = template | chat_model | parser
chain.invoke({})
# {'name': 'Samantha Smith', 'age': 29, 'city': 'New York City'}


## Structured Output Parser
schema = [
    ResponseSchema(name='Fact 1',description="Fact 1 about the topic"),
    ResponseSchema(name='Fact 2',description="Fact 2 about the topic"),
    ResponseSchema(name='Fact 3',description="Fact 3 about the topic")
]
llparser = StructuredOutputParser.from_response_schemas(schema)
template = PromptTemplate(
    template='Give me 3 facts about {topic}.\n {format_instructions}',
    input_variables = ['topic'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)
chain = template | chat_model | parser
chain.invoke({'topic':'cricket'})
{'Fact 1': 'Cricket is a popular sport played in many countries around the world.', 'Fact 2': 'The game is played between two teams of 11 players each.', 'Fact 3': 'Cricket matches can last several hours to several days, depending on the format of the game.'}


### Pydantic Output Parser
class Person(BaseModel):
    name : str = Field(description='Name of the person')
    age : int = Field(gt=18 , description= 'Age of the person')
    city : str = Field(description = 'Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Give me name, Age and city of the {country} person.\n {format_instructions}',
    input_variables= ['country'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)
chain = template | chat_model | parser
person = chain.invoke("srilanka")