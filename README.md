# CardPilot App (Navigate your contacts with AI)


This project is a **Business Card Extraction and Search App** application that allows users to extract structured information from business card images using Google's advanced multimodal AI capabilities. The extracted data, along with generated text embeddings, is stored in a **ChromaDB** vector database, enabling powerful semantic search functionality.

## Features

-   **AI-Powered Extraction**: Leverages the **Google Gemini API** (`gemini-2.0-flash` model) for multimodal analysis to extract detailed information from business cards, including names, job titles, companies, contact details, digital presence, and contextual summaries (like industry and seniority estimates).
-   **Vector Embeddings**: Uses the **Google Gemini API** (`gemini-embedding-exp-03-07` model) specialized for `retrieval_document` tasks to create semantic embeddings of the extracted business card information.
-   **Vector Database Storage**: Stores business card data and their embeddings in a persistent **ChromaDB** database (`./business_card_db`) for efficient retrieval.
-   **Semantic Search**: Enables users to search for business cards using natural language queries (e.g., "Find me AI engineers in California").
-   **Gradio Interface**: Provides an intuitive web-based interface (`app.py`) for adding new business cards (via image upload) and searching the database.
-   **Standalone Testing**: Includes a script (`src/main.py`) for testing the extraction process independently from the UI.

## Tech Stack

-   **AI Models**: Google Gemini API (Multimodal Analysis & Text Embedding)
-   **Vector Database**: ChromaDB
-   **Programming Language**: Python
-   **Web Framework**: Gradio
-   **Environment Management**: Conda

---

## Installation

Follow these steps to set up the project on your local machine:

### 1. Clone the Repository

```bash
git clone <repository-url> # Replace <repository-url> with the actual URL
cd <repository-directory> # Replace <repository-directory> with the folder name
```

### 2. Set Up the Conda Environment

Ensure you have **Anaconda** or **Miniconda** installed. Create and activate the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate cardpilot
```
This command installs all necessary Python packages and dependencies, including `google-generativeai`, `chromadb`, `gradio`, and others listed in the file.

### 3. Configure Environment Variables

Create a `.env` file in the project root directory and add your Google Gemini API key:

```env
GOOGLE_GENAI_API_KEY='YOUR_GOOGLE_GEMINI_API_KEY'
```

Replace `YOUR_GOOGLE_GEMINI_API_KEY` with your actual API key.

---

## Project Structure

```
<repository-directory>/
├── src/
│   ├── business_card_processor.py # Handles image loading and Gemini multimodal extraction
│   ├── search_engine.py           # Manages ChromaDB, embeddings, and search logic
│   ├── main.py                    # Standalone script for testing extraction
│   └── app.py                     # Gradio web interface application
├── business_card_db/              # Directory for persistent ChromaDB data
├── environment.yml                # Conda environment configuration
├── .env                           # Environment variables (API Key - create manually)
├── requirements.txt               # Pip requirements (often included for reference/alternative setup)
└── README.md                      # This file
```

*(Note: The `business_card_db/` directory will be created automatically by ChromaDB when the application runs for the first time.)*

---

## Key Components

### 1. `BusinessCardProcessor` (`src/business_card_processor.py`)
-   Loads business card images from local paths or URLs.
-   Uses the Google Gemini API (`gemini-2.0-flash`) with a detailed prompt to perform multimodal analysis, extracting structured information (including inferred details like industry and seniority) into a predefined JSON format.
-   Handles image preprocessing (encoding, hashing).

### 2. `BusinessCardVectorDB` (`src/search_engine.py`)
-   Initializes and manages a persistent ChromaDB vector collection named `business_cards`.
-   Utilizes a custom `GeminiEmbeddingFunction` that calls the Google Gemini API (`gemini-embedding-exp-03-07`) to generate embeddings for document chunks derived from the extracted card data.
-   Adds processed business card information (text chunks, metadata including the original extracted JSON, image hash) and their embeddings to the database.
-   Implements the `search_cards` method to perform semantic searches against the vector database using user queries, returning ranked results with relevance scores (distances). It also includes a `get_card_by_id` method.

### 3. Gradio Interface (`app.py`)
-   Located in the project root directory.
-   Provides a user-friendly web UI built with Gradio.
-   **"Add Business Card" Tab**: Allows users to upload a business card image. It uses `BusinessCardProcessor` and `BusinessCardVectorDB` to extract info, store it, and displays the formatted extracted details.
-   **"Search Business Cards" Tab**: Allows users to enter semantic queries and specify the number of results. It uses `BusinessCardVectorDB.search_cards` to find and display relevant results formatted with relevance scores.

### 4. Standalone Test Script (`src/main.py`)
-   A simple script to test the `BusinessCardProcessor`.

---

## Usage

### 1. Testing the Extraction Process

You can test the extraction logic directly using `main.py`:

```bash
python src/main.py
```

This script will process the image specified within it (default: `ajish_bcard.png` - ensure this file exists or change the path in `main.py`) and save the extracted data to `extracted_business_card.json` in the project root.

### 2. Running the Gradio App

Launch the Gradio web interface by running `app.py` from the project root directory:

```bash
python app.py
```

The application would then be accessible via a local URL (usually `http://127.0.0.1:7860` or `http://0.0.0.0:7860`) displayed in the terminal.

-   **Add a Business Card**: Navigate to the "Add Business Card" tab, upload an image (or use the provided examples), and click "Add to Database". The extracted information will be displayed.
-   **Search Business Cards**: Navigate to the "Search Business Cards" tab, enter your query (or use the examples), adjust the number of results if needed, and click "Search". Matching results will be displayed below.

---

## Development Notes

-   **API Key**: Ensure the `GOOGLE_GENAI_API_KEY` is correctly set in the `.env` file and that the key has access to the required Gemini models.
-   **ChromaDB Path**: The database is stored locally in the `./business_card_db` directory relative to where you run the Python scripts.
-   **Debugging**: Check the terminal output for logs and error messages from the scripts, Gemini API calls, or ChromaDB operations.
-   **Sample Image**: The `main.py` script currently points to `ajish_bcard.png`. Make sure this image file exists in the root directory or update the path in `src/main.py`.

---

## Dependencies

Key dependencies are managed via the `environment.yml` file. Major libraries include:

-   `google-generativeai`: For interacting with the Google Gemini API.
-   `chromadb`: For the vector database.
-   `gradio`: For the web interface (if `app.py` is used).
-   `python-dotenv`: For managing environment variables.
-   `Pillow`: For image handling (likely used by Gradio or potentially the processor).
-   `requests`: For fetching images from URLs.

Refer to `environment.yml` for the complete list and specific versions.

---

## Future Enhancements

-   Improve the visual formatting of the extracted information in the Gradio UI.
-   Enhance error handling and provide more specific user feedback in the UI.
-   Optimize search query processing or embedding strategy for better relevance.
-   Add options for editing or deleting existing entries in the database via the UI.
-   Integrate with cloud storage (e.g., S3 for images, managed vector DB) for scalability.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

## Acknowledgments

-   **Google** for the powerful Gemini API.
-   **ChromaDB** developers for the efficient vector database.
-   **Gradio** team for simplifying the creation of ML web interfaces.