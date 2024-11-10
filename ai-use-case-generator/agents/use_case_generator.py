from chromadb import PersistentClient
from langchain_openai import ChatOpenAI
from typing import Dict, List
import asyncio
from chromadb import PersistentClient
from langchain_openai import ChatOpenAI
from . import CompanyAnalysis, UseCase

class EnhancedUseCaseGenerator:
    def __init__(self, config: Dict):
        self.llm = ChatOpenAI(model="gpt-4o")
        self.db = PersistentClient(path=config["chroma_path"])
        self.collection = self.db.get_or_create_collection("use_cases")

    async def generate_use_cases(self, analysis: CompanyAnalysis) -> List[UseCase]:

        use_cases = await self._generate_initial_use_cases(analysis)
        
        scored_use_cases = await self._score_use_cases(use_cases, analysis)
        
        self._store_use_cases(analysis.company_name, scored_use_cases)
        
        return scored_use_cases

    async def _score_use_cases(self, use_cases: List[UseCase], analysis: CompanyAnalysis) -> List[UseCase]:

        for use_case in use_cases:
            use_case.priority_score = await self._calculate_priority_score(
                use_case, analysis
            )
        
        return sorted(use_cases, key=lambda x: x.priority_score, reverse=True)