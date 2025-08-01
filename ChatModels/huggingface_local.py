from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs={
        'temperature':0.5,
        'max_new_tokens':100
    }
)

model = ChatHuggingFace(llm=llm)
