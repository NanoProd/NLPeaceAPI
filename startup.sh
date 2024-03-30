#!/bin/bash
# Download SpaCy model
python -m spacy download en_core_web_sm

# Start the web server
gunicorn -w 1 -k uvicorn.workers.UvicornWorker app:app --pythonpath=./NLP
