from dotenv import load_dotenv
import os

load_dotenv()

CONFIG = {
    "tavily_api_key": os.getenv("TAVILY_API_KEY"),
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "chroma_path": "./chroma_db",
    "github_api": os.getenv("GITHUB_API_KEY"),
    "kaggle_api": os.getenv("KAGGLE_API_KEY"),
    "model_name": "gpt-4-turbo-preview",
    "logging_level": "INFO"
}