import weaviate, json
from weaviate.classes.config import Configure, Property, DataType
from weaviate.collections.classes.config_vectorizers import VectorDistances


client = weaviate.connect_to_embedded(version="1.30.5")
print(f"YESSSSSSS{client.is_ready()}")

# Create a collection with the vectorizer configured to use cosine distance
client.collections.delete("MyCollection")

data = [
   {
      "title": "First Object",
      "foo": 99, 
      "vector": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
   },
   {
      "title": "Second Object",
      "foo": 77, 
      "vector": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
   },
   {
      "title": "Third Object",
      "foo": 55, 
      "vector": [0.3, 0.1, -0.1, -0.3, -0.5, -0.7]
   },
   {
      "title": "Fourth Object",
      "foo": 33, 
      "vector": [0.4, 0.41, 0.42, 0.43, 0.44, 0.45]
   },
   {
      "title": "Fifth Object",
      "foo": 11,
      "vector": [0.5, 0.5, 0, 0, 0, 0]
   },
]

collection = client.collections.create(
    name="MyCollection",
    vector_config=Configure.Vectors.text2vec_openai(
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE  # Use the enum for cosine distance
        )
    ),
    properties=[
        Property(
            name="key",
            data_type=DataType.TEXT
        ),
    ],
)

print("Collection created successfully:", collection)

#Configure batch import with fixed size of 10
with client.batch.fixed_size(10) as batch:
    # Add objects to the collection in batch

    for item in data:
        properties = {
            "title": item["title"],
            "foo": item["foo"]
        }

         # Add the data object to the batch
        batch.add_object(
            collection="MyCollection",  # Name of the collection
            uuid=None,  # Automatically generate a UUID
            properties=properties,  # Properties of the object
            vector = item["vector"],  # your vector embeddings
        )


collection = client.collections.get("MyCollection")


response = collection.aggregate.over_all(total_count=True)

print("Total count:", response.total_count)

collection = client.collections.get("MyCollection")

response = (
    collection.query
    .near_vector([-0.012, 0.021, -0.23, -0.42, 0.5, 0.5])  # pass vector as positional arg
    .filter({
        "path": ["foo"],
        "operator": "GreaterThan",
        "valueNumber": 44
    })
    .with_limit(2)
    .with_additional(["distance", "id"])
    .with_properties(["title", "foo"])
    .do()  # executes the query and returns result
)

# The result is returned immediately, no .do()
print(json.dumps(response.objects, indent=2))