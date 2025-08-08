import weaviate, os, json
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

def json_print(data):
    print(json.dumps(data, indent=2))

# Connect to Weaviate Cloud using API key
auth_config = weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))
client = weaviate.connect_to_cloud(
    url=os.getenv("WEAVIATE_URL"),
    auth_client_secret=auth_config,
    additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)
print("Weaviate client connected successfully.")

client.is_ready()

print(json.dumps(client.query.aggregate("Wikipedia").with_meta_count().do(), indent=2)) # prints the number of vectors stored in this database

# Query the Wikipedia collection for vacation spots in California
response = (client.query
            .get("Wikipedia",['text','title','url','views','lang'])
            .with_near_text({"concepts": "Vacation spots in california"})
            .with_limit(5)
            .do()
           )

json_print(response)

# multilingual query for vacation spots in California
response = (client.query
            .get("Wikipedia",['text','title','url','views','lang'])
            .with_near_text({"concepts": "Miejsca na wakacje w Kalifornii"})
            .with_where({
                "path" : ['lang'],
                "operator" : "Equal",
                "valueString":'en'
            })
            .with_limit(3)
            .do()
           )

json_print(response)

# multilingual query for vacation spots in California in Arabic
response = (client.query
            .get("Wikipedia",['text','title','url','views','lang'])
            .with_near_text({"concepts": "أماكن العطلات في كاليفورنيا"})
            .with_where({
                "path" : ['lang'],
                "operator" : "Equal",
                "valueString":'en'
            })
            .with_limit(3)
            .do()
           )

json_print(response)

# rag is being used to generate a Facebook ad using the Wikipedia data
prompt = "Write me a facebook ad about {title} using information inside {text}"
result = (
  client.query
  .get("Wikipedia", ["title","text"])
  .with_generate(single_prompt=prompt)
  .with_near_text({
    "concepts": ["Vacation spots in california"]
  })
  .with_limit(3)
).do()

json_print(result)

generate_prompt = "Summarize what these posts are about in two paragraphs."
 # Use a single prompt to summarize all posts
result = (
  client.query
  .get("Wikipedia", ["title","text"])
  .with_generate(grouped_task=generate_prompt) # Pass in all objects at once
  .with_near_text({
    "concepts": ["Vacation spots in california"]
  })
  .with_limit(3)
).do()

json_print(result)