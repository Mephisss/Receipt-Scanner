# Receipt Scanner

Data extractor from pictures of store receipts

## Setup

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Get Groq API key**
- Sign up at https://console.groq.com
- Get free API key from https://console.groq.com/keys

3. **Create `.env` file**
```
GROQ_API_KEY=your_api_key_here
```

## Run
```bash
python app.py
```

Open http://localhost:5000

## Files

- `model_llm.py` - LLM vision extractor
- `app.py` - Flask web server
- `templates/index.html` - Web interface


## Requirements

- Python 3.8+
- Groq API key (free tier works)
- Internet connection (for API calls)
