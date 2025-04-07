# Retrieval-Augmented Generation (RAG) Pipeline

Welcome to the RAG Pipeline repository! This project demonstrates the implementation of Retrieval-Augmented Generation (RAG) systems, starting from basic setups using Ollama, NumPy, and JSON, to more advanced configurations employing LangChain, Ollama, HuggingFace, and FAISS for enhanced performance.

![image](https://github.com/user-attachments/assets/04b344ac-7c1c-432c-b383-6b5c24c9aa7b)


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Basic RAG with Ollama, NumPy, and JSON](#basic-rag-with-ollama-numpy-and-json)
  - [Advanced RAG with LangChain, Ollama, HuggingFace, and FAISS](#advanced-rag-with-langchain-ollama-huggingface-and-faiss)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

Retrieval-Augmented Generation (RAG) combines information retrieval with text generation, enabling language models to fetch and incorporate relevant external information into their responses. This approach enhances the accuracy and relevance of generated content, especially when dealing with topics beyond the model's original training data.

## Features

- **Basic RAG Implementation**: Utilize Ollama, NumPy, and JSON for foundational RAG functionalities.
- **Advanced RAG Setup**: Leverage LangChain, Ollama, HuggingFace, and FAISS for optimized retrieval and generation processes.
- **Document Embedding and Storage**: Efficiently embed documents and store them in vector databases for quick retrieval.
- **Interactive Query Handling**: Process user queries by retrieving pertinent information and generating informed responses.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- [Ollama](https://ollama.ai/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [LangChain](https://python.langchain.com/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Rktim/RAG.git
   cd RAG
   ```

2. **Set Up a Virtual Environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```

3. **Install Required Packages**:

   ```bash
   pip install -r requirements.txt
   ```

   *Note*: If `requirements.txt` is not provided, manually install the necessary packages:

   ```bash
   pip install numpy ollama faiss-cpu langchain transformers
   ```

## Usage

### Basic RAG with Ollama, NumPy, and JSON

This setup demonstrates a fundamental RAG system using Ollama for language modeling, NumPy for numerical operations, and JSON for data handling.

1. **Prepare Your Data**: Ensure your documents are available in JSON format.

2. **Run the Basic RAG Script**:

   ```bash
   python basic_rag.py
   ```

   This script will:

   - Load documents from the JSON file.
   - Generate embeddings using Ollama.
   - Store embeddings and documents.
   - Retrieve relevant documents based on user queries.
   - Generate responses incorporating retrieved information.

### Advanced RAG with LangChain, Ollama, HuggingFace, and FAISS

For enhanced performance, this setup integrates LangChain for orchestration, Ollama for language modeling, HuggingFace for embeddings, and FAISS for efficient similarity search.

1. **Prepare Your Documents**: Place your documents in the designated directory (e.g., `data/`).

2. **Run the Advanced RAG Script**:

   ```bash
   python advanced_rag.py
   ```

   This script will:

   - Load and preprocess documents.
   - Split documents into manageable chunks.
   - Generate embeddings using HuggingFace models.
   - Store embeddings in a FAISS vector database.
   - Retrieve relevant document chunks based on user queries.
   - Generate responses using Ollama, incorporating retrieved information.

## Project Structure

```
RAG/
├── data/                         # Directory containing source documents
├── embeddings/                   # Directory for storing embeddings and vector databases
├── basic_rag.py                  # Script for basic RAG implementation
├── advanced_rag.py               # Script for advanced RAG implementation
├── requirements.txt              # List of dependencies
├── LICENSE                       # License information
└── README.md                     # This README file
```

- **`data/`**: Contains source documents used for retrieval.
- **`embeddings/`**: Stores generated embeddings and FAISS index files.
- **`basic_rag.py`**: Implements the basic RAG system.
- **`advanced_rag.py`**: Implements the advanced RAG system with enhanced capabilities.
- **`requirements.txt`**: Lists all Python packages required to run the project.
- **`LICENSE`**: Contains the MIT License under which this project is distributed.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with clear descriptions.
4. Push your branch and submit a pull request.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

Special thanks to the developers and contributors of [Ollama](https://ollama.ai/), [FAISS](https://github.com/facebookresearch/faiss), [LangChain](https://python.langchain.com/), and [HuggingFace Transformers](https://huggingface.co/transformers) for providing the foundational tools that made this project possible.
