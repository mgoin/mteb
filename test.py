from mteb import MTEB

model_name = "TaylorAI/bge-micro-v2"

from deepsparse_sentence_transformers import DeepSparseSentenceTransformer
model = DeepSparseSentenceTransformer(model_name_or_path=model_name, export=True)
evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(model, output_folder=f"results/ds-{model_name}")
print(results)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer(model_name)
evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(model, output_folder=f"results/st-{model_name}")
print(results)
