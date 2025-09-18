# Clarus
A new paradigm for technical writing

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   - Copy `env.example` to `.env`:
     ```bash
     cp env.example .env
     ```
   - Edit `.env` and add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_actual_openai_api_key_here
     ```

3. **Run the application:**
   ```bash
   streamlit run streamlit_app.py
   ```

## Environment Variables

The application loads environment variables from a `.env` file in the project root. Required variables:

- `OPENAI_API_KEY`: Your OpenAI API key for LLM functionality

Optional variables:
- `ANTHROPIC_API_KEY`: For Anthropic models (if used)
- `LANGCHAIN_API_KEY`: For LangChain services (if used)