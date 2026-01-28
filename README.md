# FinoBot 📈: Financial News & Analysis

<div align="center">
  <img src="New_logo.png" alt="FinoBot Logo" width="400"/>
  
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
  [![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://www.langchain.com/)
  [![OpenAI](https://img.shields.io/badge/OpenAI-API-orange.svg)](https://openai.com/)
</div>

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

**FinoBot** is an intelligent AI-powered research assistant designed to analyze financial news articles and provide insightful answers to your queries. Built with Streamlit and LangChain, it leverages advanced natural language processing to extract, process, and answer questions from financial news URLs.

## ✨ Features

- 📰 **URL-Based Research**: Analyze multiple financial news articles by simply providing URLs
- 🤖 **AI-Powered Q&A**: Ask questions and get accurate answers with source citations
- 💾 **Vector Database**: Uses FAISS for efficient semantic search and retrieval
- 📝 **Session Management**: Maintains chat history and processed URLs across sessions
- 🔍 **Source Attribution**: Always provides sources for answers with direct links
- 📊 **Logging System**: Comprehensive logging with rotating file handlers
- 🎨 **Modern UI**: Clean, intuitive interface with custom styling

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Streamlit**: Web application framework
- **LangChain**: LLM orchestration and chaining
- **OpenAI GPT**: Language model for understanding and generation
- **FAISS**: Vector database for similarity search
- **SQLite**: Local database for session and history management

### Key Libraries
- `langchain-openai`: OpenAI integration with LangChain
- `langchain-community`: Community tools and document loaders
- `sentence-transformers`: Text embeddings
- `unstructured`: Document parsing
- `nltk`: Natural language processing
- `beautifulsoup4`: Web scraping (via unstructured)
- `faiss-cpu`: Vector similarity search
- `python-dotenv`: Environment variable management

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- OpenAI API key

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Varsha-1605/Research-Analyst.git
   cd Research-Analyst
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy the sample environment file
   cp .env_sample .env
   
   # Edit .env and add your OpenAI API key
   # OPENAI_API_KEY=your_api_key_here
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory with the following:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Streamlit Secrets (for deployment)

For Streamlit Cloud deployment, add your API key to `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
```

## 🚀 Usage

### Running Locally

1. **Start the application**
   ```bash
   streamlit run final_research_analyst.py
   ```

2. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in your terminal

### Using the Application

1. **Enter URLs**: In the sidebar, input URLs of financial news articles (up to 3 URLs)
2. **Process URLs**: Click the "Process URLs" button to analyze the content
3. **Ask Questions**: Type your question in the main chat interface
4. **Get Answers**: Receive AI-generated answers with source citations
5. **View History**: Previous conversations are maintained in the session

### Example Queries
- "What are the key financial highlights mentioned in the articles?"
- "What is the stock performance discussed?"
- "Summarize the main points about the company's earnings."
- "What are the analyst predictions mentioned?"

## 📁 Project Structure

```
Research-Analyst/
│
├── final_research_analyst.py    # Main application file
├── requirements.txt             # Python dependencies
├── .env_sample                  # Sample environment variables
├── .env                         # Your environment variables (not tracked)
├── .gitignore                   # Git ignore rules
├── New_logo.png                 # Application logo
├── README.md                    # This file
│
├── logs/                        # Application logs directory
│   └── finobot.log             # Rotating log files
│
└── finobot.db                   # SQLite database (auto-generated)
```

## 🔧 How It Works

### Architecture Overview

```
User Input (URLs) → Document Loader → Text Splitter → 
Embeddings → FAISS Vector Store → RetrievalQA Chain → 
LLM (GPT) → Answer with Sources
```

### Process Flow

1. **Document Loading**: URLs are loaded using `UnstructuredURLLoader`
2. **Text Splitting**: Documents are split into chunks using `RecursiveCharacterTextSplitter`
3. **Embedding Generation**: Text chunks are converted to embeddings using `OpenAIEmbeddings`
4. **Vector Storage**: Embeddings are stored in FAISS for fast retrieval
5. **Query Processing**: User questions are embedded and matched against stored vectors
6. **Answer Generation**: Relevant chunks are sent to GPT for answer generation
7. **Source Citation**: Answers include references to source URLs

### Database Schema

**processed_urls**
- `url` (TEXT, PRIMARY KEY): Processed URL
- `session_id` (TEXT): Associated session identifier

**chat_history**
- `id` (INTEGER, PRIMARY KEY): Message ID
- `role` (TEXT): User or assistant
- `message` (TEXT): Message content
- `session_id` (TEXT): Session identifier

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comments for complex logic
- Update documentation for new features
- Test thoroughly before submitting PR

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **OpenAI** for GPT models
- **LangChain** for the excellent framework
- **Streamlit** for the web framework
- **FAISS** by Meta AI for vector search

## 📧 Contact

**Varsha** - [@Varsha-1605](https://github.com/Varsha-1605)

Project Link: [https://github.com/Varsha-1605/Research-Analyst](https://github.com/Varsha-1605/Research-Analyst)

---

<div align="center">
  Made with ❤️ for financial research and analysis
</div>