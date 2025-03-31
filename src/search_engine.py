import os
import json
import chromadb

# from chromadb.errors import CollectionNotFoundError
# import base64
import hashlib
import pathlib
import requests
from chromadb.api.types import (
    EmbeddingFunction,
    Embeddings,
    Metadata,
    Include,
    QueryResult,
    IncludeEnum,
)
from typing import List, Dict, Any, Sequence, Tuple, Literal
from google import genai
from google.genai import types
from dotenv import load_dotenv
from .business_card_processor import BusinessCardProcessor

# Load environment variables
load_dotenv()


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def __call__(self, input: Sequence[str]) -> Embeddings:
        embeddings = []
        for text in input:
            # Use retrieval_document for storing documents
            result = self.client.models.embed_content(
                model="gemini-embedding-exp-03-07",
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="retrieval_document", title="Business Card Information"
                ),
            )
            if (
                result.embeddings
                and hasattr(result.embeddings[0], "values")
                and result.embeddings[0].values is not None
            ):
                # Extract just the values from the embedding response
                values = result.embeddings[0].values
                embeddings.append([float(x) for x in values])
            else:
                embeddings.append([0.0])  # Fallback for failed embeddings
        return embeddings


class BusinessCardVectorDB:
    def __init__(self):
        # Configure Gemini
        self.api_key = os.getenv("GOOGLE_GENAI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_GENAI_API_KEY environment variable not set")

        # Initialize processors
        self.processor = BusinessCardProcessor()
        self.client = genai.Client(api_key=self.api_key)

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="./business_card_db")

        # Initialize embedding function
        embedding_function = GeminiEmbeddingFunction(self.api_key)

        # List all collections
        collections = self.chroma_client.list_collections()
        exists = "business_cards" in collections

        if exists:
            print("Using existing collection 'business_cards'")
            self.collection = self.chroma_client.get_collection(
                name="business_cards", embedding_function=embedding_function
            )
        else:
            print("Creating new collection 'business_cards'")
            self.collection = self.chroma_client.create_collection(
                name="business_cards",
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"},
            )

    def load_image(self, image_source: str) -> Tuple[bytes, str]:
        """Load image from URL or local path and return bytes with hash."""
        try:
            if image_source.startswith(("http://", "https://")):
                print("Loading image from URL...")
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                image_bytes = response.content
            else:
                print("Loading image from local path...")
                image_bytes = pathlib.Path(image_source).read_bytes()

            return image_bytes, hashlib.sha256(image_bytes).hexdigest()
        except Exception as e:
            raise RuntimeError(f"Error loading image: {e}")

    def process_llm_response(self, response) -> Dict[str, Any]:
        """Process API response and return extracted data."""
        if not response or not response.text:
            print("No valid response received from the API.")
            return {}

        try:
            cleaned_data = response.text.strip()
            # Handle various JSON response formats
            for wrapper in ["```json", "```"]:
                if cleaned_data.startswith(wrapper) and cleaned_data.endswith("```"):
                    cleaned_data = cleaned_data[len(wrapper) : -3].strip()
            return json.loads(cleaned_data)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}\nRaw response: {response.text}")
        except Exception as e:
            print(f"Unexpected error processing response: {e}")
        return {}

    def extract_from_image(self, image_source: str) -> str:
        """Extract business card information directly from image and add to database."""
        # Use the processor to extract information
        card_data = self.processor.extract_from_image(image_source)

        # Add to vector database
        return self.add_business_card(card_data)

    def process_business_card(self, card_json: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and structure information from the JSON"""
        extracted = card_json.get("extracted_info", {})
        primary = extracted.get("primary_info", {})
        contact = extracted.get("contact_info", {})
        digital = extracted.get("digital_presence", {})
        contextual = extracted.get("contextual_summary", {})

        documents = [
            # Professional identity
            f"{primary['name']['value']} is a {primary['job_title']['value']} in {contextual.get('industry_inference', '')}based in {contact.get('addresses', [{}])[0].get('value', '')}.",
            # Contact information
            f"Contact: {contact.get('emails', [{}])[0].get('value', '')} | Location: {contact.get('addresses', [{}])[0].get('value', '')} | Website: {digital.get('website', {}).get('value', '')}",
            # Digital presence
            f"Social media: {', '.join([f'{sm["platform"]}:{sm["handle"]}' for sm in digital.get('social_media', [])])}",
            # Visual description
            f"professional_summary: {contextual.get('professional_summary', '')} expertise in {contextual.get('industry_inference', '')} and {contextual.get('seniority_estimate', '')}",
        ]

        # Prepare metadata - ensure all values are strings or simple types
        metadata: Metadata = {
            "name": str(primary["name"]["value"]),
            "job_title": str(primary["job_title"]["value"]),
            "company": str(primary["company"].get("text_value", "")),
            "email": str(contact.get("emails", [{}])[0].get("value", "")),
            "location": str(contact.get("addresses", [{}])[0].get("value", "")),
            "website": str(digital.get("website", {}).get("value", "")),
            "industry": str(contextual.get("industry_inference", "")),
            "seniority": str(contextual.get("seniority_estimate", "")),
            "image_hash": str(card_json["image_metadata"]["hash"]),
            "source_json": json.dumps(
                card_json
            ),  # Store the original JSON for later retrieval
        }

        return {
            "documents": documents,
            "metadata": metadata,
            "image_base64": card_json["image_metadata"]["base64"],
        }

    def add_business_card(self, card_json: Dict[str, Any]) -> str:
        """Add a business card to the vector database"""
        processed = self.process_business_card(card_json)

        # Generate an ID based on name and hash
        card_id = (
            f"{processed['metadata']['name']}-{processed['metadata']['image_hash']}"
        )

        # Add to ChromaDB collection
        self.collection.add(
            documents=processed["documents"],
            metadatas=[processed["metadata"]]
            * len(processed["documents"]),  # Same metadata for all docs
            ids=[
                f"{card_id}_0",
                f"{card_id}_1",
                f"{card_id}_2",
                f"{card_id}_3",
            ],  # Unique IDs for each doc
        )

        return card_id

    def process_query(self, query: str) -> str:
        """Process and enhance the search query for better retrieval."""
        try:
            # Create a more comprehensive search query
            enhanced_query = f"""
            Find business cards matching the following criteria:
            {query}
            Consider job titles, names, locations, and professional context.
            """
            print(f"Processing search query: {enhanced_query}")
            return enhanced_query
        except Exception as e:
            print(f"Query processing error: {e}")
            return query

    def search_cards(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search for business cards matching the query."""
        try:
            processed_query = self.process_query(query)
            print(f"Searching with processed query: {processed_query}")

            # Get the total count of documents in the collection
            all_ids = self.collection.get()["ids"]
            if not all_ids:
                print("No documents in collection")
                return []

            print(f"Total documents in collection: {len(all_ids)}")

            # Perform the search
            results = self.collection.query(
                query_texts=[processed_query],
                n_results=min(
                    num_results * 4, len(all_ids)
                ),  # Request more results to account for duplicates
                include=["documents", "metadatas", "distances"],  # type: ignore
            )

            if not results or not results.get("ids"):
                print("No results found")
                return []

            print(f"Raw results found: {len(results['ids'][0])}")

            # Process results
            formatted_results = []
            seen_cards = set()  # Track unique cards

            for i in range(len(results["ids"][0])):
                card_id = results["ids"][0][i].rsplit("_", 1)[0]  # Remove _N suffix

                # Skip if we've already seen this card
                if card_id in seen_cards:
                    continue
                seen_cards.add(card_id)

                card_data = self.get_card_by_id(card_id)
                if card_data and "metadata" in card_data:
                    try:
                        extracted_info = json.loads(
                            str(card_data["metadata"].get("source_json", "{}"))
                        )
                        distances = results.get("distances", [])
                        distance = (
                            distances[0][i]
                            if distances
                            and len(distances) > 0
                            and len(distances[0]) > i
                            else 1.0
                        )
                        result = {
                            "metadata": card_data["metadata"],
                            "distance": distance,
                            "extracted_info": extracted_info.get("extracted_info", {}),
                        }
                        formatted_results.append(result)

                        if len(formatted_results) >= num_results:
                            break
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON for card {card_id}")
                        continue

            print(f"Returning {len(formatted_results)} unique results")
            return formatted_results

        except Exception as e:
            print(f"Search error: {e}")
            return []

    def get_card_by_id(self, card_id: str) -> Dict[str, Any]:
        """Retrieve a complete business card by its ID"""
        results = self.collection.get(
            ids=[f"{card_id}_0", f"{card_id}_1", f"{card_id}_2", f"{card_id}_3"],
            include=["metadatas"],  # type: ignore
        )

        if not results["metadatas"]:
            return {
                "metadata": {},
                "original_json": "{}",
                "image_base64": "",
            }

        # Return the first metadata (they're all the same)
        metadata = results["metadatas"][0]

        # Parse the stored original JSON
        original_json = json.loads(str(metadata.get("source_json", "{}")))

        return {
            "metadata": metadata,
            "original_json": original_json,
            "image_base64": metadata.get("image_base64", ""),
        }


# Example usage
if __name__ == "__main__":
    # Initialize the vector DB
    db = BusinessCardVectorDB()

    # Process business card directly from image
    image_path = "input/ak.jpeg"  # or a URL
    card_id = db.extract_from_image(image_path)
    print(f"Added business card with ID: {card_id}")

    # Example search
    query = "desinger in africa"
    results = db.search_cards(query)
    print(f"\nSearch results for '{query}':")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Name: {result['metadata']['name']}")
        print(f"Title: {result['metadata']['job_title']}")
        print(f"Location: {result['metadata']['location']}")
        print(f"Relevance score: {1 - result['distance']:.2f}")
        print("Matching text snippets:")
        for doc in result["documents"]:
            print(f"- {doc[:100]}...")

    # Retrieve full card by ID
    full_card = db.get_card_by_id(card_id)
    if full_card:
        print("\nFull card retrieval:")
        print(f"Name: {full_card['metadata']['name']}")
        print(f"Image available: {len(full_card.get('image_base64', '')) > 0}")
