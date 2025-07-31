import weaviate, json
from weaviate import EmbeddedOptions

client = weaviate.Client(embedded_options=EmbeddedOptions(),)
client.is_ready()