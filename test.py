from mteb import MTEB

model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

from deepsparse_sentence_transformers import DeepSparseSentenceTransformer
model = DeepSparseSentenceTransformer(model_name_or_path=model_name, export=True)
evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(model, output_folder=f"results/{model_name}")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer(model_name)
evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(model, output_folder=f"results/{model_name}")