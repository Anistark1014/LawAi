git clone [https://github.com/AmarJogdand/AI-Justice_Bot.git](https://github.com/AmarJogdand/AI-Justice_Bot.git)
# Simple Legal AI API

The Simple Legal AI API is a lightweight retrieval-based assistant that serves practical legal guidance derived from a curated collection of Indian law documents. The project exposes a Flask API backed by FAISS and sentence-transformer embeddings so that clients can request legal advice, perform semantic search, and monitor service health from any frontend.

## Project Highlights

- Retrieval-first architecture using `FAISS` and `sentence-transformers` (`paraphrase-multilingual-mpnet-base-v2`).
- 6,611 pre-indexed legal document chunks stored in `simple_legal_model/` (embeddings, FAISS index, and metadata).
- REST API built with Flask and CORS support for browser clients.
- Example React hooks and components under `react_example/` for web integration.
  
## Repository Layout

```
LawAi/
├── simple_legal_api.py        # Flask application with retrieval + advice logic
├── start_api.py               # Convenience launcher for local development
├── simple_legal_model/        # Pre-built embeddings and FAISS index
├── react_example/             # Sample frontend integration code
├── requirements.txt           # Python dependencies
├── templates/                 # HTML templates if needed for flask.render_template()
└── ...                        # Additional experimental and archived scripts
```

## Prerequisites

- Windows, macOS, or Linux
- Python 3.10+ (project tested on Python 3.12)
- (Optional) Virtual environment tooling such as `venv` or `conda`

## Setup

```powershell
git clone https://github.com/Anistark1014/LawAi.git
cd LawAi

# (Optional) create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

The repository already ships with a trained retrieval model in `simple_legal_model/`. If you ever delete it, regenerate the files with:

```powershell
python simple_train.py
```

## Running the API Server

```powershell
python start_api.py
```

`start_api.py` loads the pre-computed model, starts the Flask server on `http://127.0.0.1:5000`, and prints startup logs. To stop the server press `Ctrl+C` in the same terminal.

### Production Notes

`start_api.py` is designed for local usage. For production deployments, prefer a WSGI container (for example, `waitress` on Windows or `gunicorn` on Linux). An example command using waitress:

```powershell
pip install waitress
python -m waitress --listen=0.0.0.0:5000 start_api:app
```

## API Endpoints

| Method | Path              | Description                           |
|--------|-------------------|---------------------------------------|
| GET    | `/api/health`     | Returns service health and model info |
| POST   | `/api/legal-advice` | Generates legal guidance for a query |
| POST   | `/api/search`     | Performs semantic search over documents |

### Example Requests

Health check:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/health" -Method GET
```

Ask for legal advice:

```powershell
$body = @{ message = "My phone was stolen in a bus. What can I do?" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/legal-advice" -Method POST -Body $body -ContentType "application/json"
```

Semantic search:

```powershell
$body = @{ query = "Section 420 punishment"; results = 5 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/search" -Method POST -Body $body -ContentType "application/json"
```

## Frontend Integration

Sample React code lives in `react_example/`:

- `useLegalAPI.js`: hook for querying the Flask API from React.
- `LegalAssistant.jsx`: basic UI for text prompts uploading is handled separately.

Import the hook into a React project and call `getLegalAdvice` or `checkHealth` to integrate the backend into web applications.

## Model Artifacts

`simple_legal_model/` contains:

- `documents.json`: metadata and text chunks used during retrieval.
- `embeddings.npy`: dense vectors for each chunk.
- `legal_index.faiss`: FAISS index matching queries to relevant chunks.
- `model_info.json`: information about the embedding model and dataset.
- `tfidf_matrix.pkl`, `tfidf_vectorizer.pkl`: optional TF-IDF fallback assets.

To refresh these artifacts with new legal content, update the source documents and run `python simple_train.py`.

## Troubleshooting

- **Model directory not found**: ensure `simple_legal_model/` exists or rerun `python simple_train.py`.
- **UnicodeEncodeError in PowerShell**: the logging output is ASCII only; if you reintroduce emoji make sure the console code page supports UTF-8.
- **Port already in use**: stop any existing Python server or change the port in `start_api.py` (`app.run(port=5001)`).
- **Slow startup**: the sentence-transformer loads a large model; the first run can take several seconds.

## License

Any upstream license terms still apply. Refer to the original repository or accompanying license files for details.
