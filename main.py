# apparently llama index can be linked with Ollama: quite interesting!
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from dotenv import load_dotenv
from prompts import context
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from code_reader import code_reader
from llama_index.core.agent import ReActAgent
import ast
import os

load_dotenv()

llm = Ollama(
    model='mistral',
    request_timeout = 30.0
)

# result = llm.complete('Hello World')
# print(result)
'''
LlamaParse: Proprietary parsing for complex documents with embedded objects such as tables and figures. LlamaParse directly integrates with LlamaIndex ingestion and retrieval to let you build retrieval over complex, semi-structured documents. You’ll be able to answer complex questions that simply weren’t possible previously.

'''
parser = LlamaParse(result_type='markdown')

# finding any pdf will be parsed with the llama parser we defined
file_extractor = {'.pdf': parser}

documents = SimpleDirectoryReader('./data', file_extractor=file_extractor).load_data()
# used to generate embeddings
embed_model = resolve_embed_model('local:BAAI/bge-m3')

vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)

# res = query_engine.query('What are some of the routes in the api?')
# print(res)

tools = [
    # one of the tool our agent can use.
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="This gives documentation about code for an API. Use this for reading docs for the API."
        )
    ),
    code_reader
]

# define a code llm which is specialized for coding tasks
code_llm = Ollama(model="codellama")
# verbose = True means agent will share their thought process
agent = ReActAgent.from_tools(tools, llm=code_llm, verborse=True, context="""Purpose: The primary role of this agent is to assist users by analyzing code. It should be able to generate code and answer questions about code provided.""")

'''Fancy way of writing this:
prompt = input("Enter a prompt (q to quit): ")  # Initial input before the loop starts
while prompt != "q":  # Loop continues as long as the user doesn't enter "q"
    prompt = input("Enter a prompt (q to quit): ")  # Prompt the user for new input on each iteration
'''

class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str




parser = PydanticOutputParser(CodeOutput)
json_prompt_str = parser.format("""Parse the response from a previous LLM into a description and a string of valid code, also come up with a valid filename this could be saved as that doesnt contain special characters. 
Here is the response: {response}. You should parse this in the following JSON Format: """)
json_prompt_tmpl = PromptTemplate(json_prompt_str)
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, llm])
# Prompt 1: Send a post request to make a new item using the api in python.
# Prompt 2: Read the contents of test.py and extract all codes to output for me.
# Prompt 3: Read the contents of test.py and write a python script to call the post endpoint to make a new item
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    # when prompt is not = 'q', we will pass that prompt to agent
    # result = agent.query(prompt)
    # print(result)

    retries = 0

    while retries < 3:
        try:
            result = agent.query(prompt)
            next_result = output_pipeline.run(response=result)
            cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))
            break
        except Exception as e:
            retries += 1
            print(f"Error occured, retry #{retries}:", e)

    if retries >= 3:
        print("Unable to process request, try again...")
        continue

    print("Code generated")
    print(cleaned_json["code"])
    print("\n\nDesciption:", cleaned_json["description"])

    filename = cleaned_json["filename"]

    try:
        with open(os.path.join("output", filename), "w") as f:
            f.write(cleaned_json["code"])
        print("Saved file", filename)
    except:
        print("Error saving file...")