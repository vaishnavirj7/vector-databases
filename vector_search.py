import weaviate, json
from weaviate import EmbeddedOptions

client = weaviate.Client(embedded_options=EmbeddedOptions(),)
client.is_ready()
if client.schema.exists("Question"):
    client.schema.delete_class("Question")
class_obj = {
    "class": "Question",
    "vectorizer": "text2vec-openai",  # Use OpenAI as the vectorizer
    "moduleConfig": {
        "text2vec-openai": {
            "model": "ada",
            "modelVersion": "002",
            "type": "text",
            "baseURL": os.environ["OPENAI_API_BASE"]
        }
    }
}

client.schema.create_class(class_obj)