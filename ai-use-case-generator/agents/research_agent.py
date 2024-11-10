import asyncio
import logging
from typing import Dict, List
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from chromadb import PersistentClient
from . import CompanyAnalysis

class EnhancedResearchAgent:
    def __init__(self, config: Dict):
        self.search_tool = TavilySearchResults()
        self.llm = ChatOpenAI(model="gpt-4o")
        self.db = PersistentClient(path=config["chroma_path"])
        self.collection = self.db.get_or_create_collection("market_research")
        self.logger = logging.getLogger(__name__)

    async def research_company(self, company: str) -> CompanyAnalysis:
        try:
            searches = await asyncio.gather(
                self._search_company_info(company),
                self._search_industry_trends(company),
                self._search_competitors(company)
            )

            analysis = await self._analyze_findings(company, searches)
            
            self._store_analysis(company, analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in research_company: {str(e)}")
            raise

    async def _analyze_findings(self, company: str, searches: List[Dict]) -> CompanyAnalysis:
        prompt = self._create_analysis_prompt(company, searches)
        response = await self.llm.ainvoke(prompt)
        
        return self._parse_analysis_response(response)