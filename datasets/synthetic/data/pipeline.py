# Requirements: `pip install distilabel[hf-inference-endpoints]`
import os
import random
from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, KeepColumns
from distilabel.steps.tasks import GenerateTextClassificationData

MODEL = "qwen2.5:32b-instruct-q5_K_S"
BASE_URL = "http://127.0.0.1:11434/"
TEXT_CLASSIFICATION_TASK = "None"
os.environ["API_KEY"] = (
    "hf_xxx"  # https://huggingface.co/settings/tokens/new?ownUserPermissions=repo.content.read&ownUserPermissions=repo.write&globalPermissions=inference.serverless.write&canReadGatedRepos=true&tokenType=fineGrained
)

with Pipeline(name="textcat") as pipeline:

    task_generator = LoadDataFromDicts(data=[{"task": TEXT_CLASSIFICATION_TASK}])

    textcat_generation = GenerateTextClassificationData(
        llm=InferenceEndpointsLLM(
            model_id=MODEL,
            base_url=BASE_URL,
            api_key=os.environ["API_KEY"],
            generation_kwargs={
                "temperature": 0.8,
                "max_new_tokens": 1024,
                "top_p": 0.95,
            },
        ),
        seed=random.randint(0, 2**32 - 1),
        difficulty='high school',
        clarity=None,
        num_generations=10,
        output_mappings={"input_text": "text"},
    )
    
    keep_columns = KeepColumns(
        columns=["text", "label"],
    )

    # Connect steps in the pipeline
    task_generator >> textcat_generation >> keep_columns

    if __name__ == "__main__":
        distiset = pipeline.run()