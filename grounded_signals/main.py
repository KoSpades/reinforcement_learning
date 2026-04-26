import requests
import gzip
import json

# This is a direct link to one chunk of the English dataset
url = "https://data.together.xyz/redpajama-data-v2/v1.0.0/documents/2023-06/0000/en_head.json.gz"

print("Fetching documents from RedPajama-V2...\n")

# Connect to the dataset
with requests.get(url, stream=True) as response:
    # Decompress the data on the fly
    with gzip.GzipFile(fileobj=response.raw) as f:
        
        # Read the first 3 documents (change this number if you want to read more)
        for i in range(3): 
            line = f.readline()
            
            # Convert the JSON line into a readable Python dictionary
            doc = json.loads(line)
            
            # Print the text so you can actually read it
            print(f"=== Document {i+1} ===")
            print(f"Source URL: {doc.get('url')}\n")
            print(doc.get('raw_content'))
            print("\n" + "="*80 + "\n")