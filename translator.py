from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.llms import CTransformers
from langserve import add_routes

system_template = "Translate the following from {source} into {destination}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

model = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML",
                      config={'max_new_tokens': 256,
                              'temperature': 0.01})

parser = StrOutputParser()
chain = prompt_template | model | parser
app = FastAPI(
    title="Translation Service",
    version="1.0",
    description="A simple Translator API server using llama2",
)

add_routes(
    app,
    chain,
    path="/translate",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
