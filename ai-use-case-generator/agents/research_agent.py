# research_agent.py

import asyncio
import logging
import json
from typing import Dict, List, Optional
from datetime import datetime
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from chromadb import PersistentClient
from . import CompanyAnalysis

class EnhancedResearchAgent:

    def __init__(self, config: Dict):
        self.search_tool = TavilySearchResults(api_key=config["tavily_api_key"])
        self.llm = ChatOpenAI(
            model=config["model_name"],
            api_key=config["openai_api_key"],
            temperature=0.1
        )
        self.db = PersistentClient(path=config["chroma_path"])
        self.collection = self.db.get_or_create_collection("market_research")
        self.logger = logging.getLogger(__name__)
    
    # Fixed prompt template with escaped curly braces
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", "You must respond with a valid JSON object only. No other text or formatting."),
            ("human", """
            Analyze the following information about {company} and create a structured analysis.
            Search results: {search_results}
        
            The response must be a single valid JSON object with this exact structure:
            {{
                "industry": "string",
                "business_model": "string",
                "key_products": ["string"],
                "market_position": "string",
                "competitors": ["string"],
                "ai_readiness": float,
                "trends": "string"
            }}
            """)
        ])

    async def research_company(self, company: str) -> Dict:
        """Execute company research workflow and return analysis as dict."""
        try:
            searches = await asyncio.gather(
                self._search_company_info(company),
                self._search_industry_trends(company),
                self._search_competitors(company)
            )
            all_searches = [item for sublist in searches for item in sublist]
            analysis = await self._analyze_findings(company, all_searches)
            self._store_analysis(company, analysis)

            return analysis.to_dict()
        except Exception as e:
            self.logger.error(f"Research error for {company}: {str(e)}")
            raise

    async def _search_company_info(self, company: str) -> List[Dict]:
        """Search for company information."""
        query = f"{company} company overview business model revenue products services"
        try:
            results = await self.search_tool.ainvoke({"query": query})
            return self._filter_results(results)
        except Exception as e:
            self.logger.error(f"Company info search error: {str(e)}")
            return []

    async def _search_industry_trends(self, company: str) -> List[Dict]:
        """Search for industry trends."""
        query = f"{company} industry market trends analysis future outlook"
        try:
            results = await self.search_tool.ainvoke({"query": query})
            return self._filter_results(results)
        except Exception as e:
            self.logger.error(f"Industry trends search error: {str(e)}")
            return []

    async def _search_competitors(self, company: str) -> List[Dict]:
        """Search for competitors."""
        query = f"{company} competitors market share comparison"
        try:
            results = await self.search_tool.ainvoke({"query": query})
            return self._filter_results(results)
        except Exception as e:
            self.logger.error(f"Competitors search error: {str(e)}")
            return []

    def _filter_results(self, results: List[Dict], min_score: float = 0.6) -> List[Dict]:
        """Filter search results by relevance score."""
        return [r for r in results if r.get("relevance_score", 0) >= min_score]

    def _clean_json_response(self, response: str) -> str:
        """Clean and validate JSON response from LLM."""
        cleaned = response.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned[3:-3].strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        return cleaned

    async def _analyze_findings(self, company: str, searches: List[Dict]) -> CompanyAnalysis:
        """Analyze search findings using LLM."""
        try:
            formatted_searches = json.dumps(searches, indent=2)
            chain = self.analysis_prompt | self.llm
            response = await chain.ainvoke({
                "company": company,
                "search_results": formatted_searches
            })
            return self._parse_analysis_response(response, company)
        except Exception as e:
            self.logger.error(f"Analysis error: {str(e)}")
            raise

    def _parse_analysis_response(self, response: AIMessage, company: str) -> CompanyAnalysis:
        """Parse LLM response into CompanyAnalysis object."""
        try:
            content = response.content
            if isinstance(content, str):
                cleaned_content = self._clean_json_response(content)
                content = json.loads(cleaned_content)

            required_fields = ["industry", "business_model", "key_products", 
                             "market_position", "competitors", "ai_readiness", "trends"]
            missing_fields = [field for field in required_fields if field not in content]
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")

            return CompanyAnalysis(
                company_name=company,
                industry=content["industry"],
                business_model=content["business_model"],
                key_products=content["key_products"],
                market_position=content["market_position"],
                competitors=content["competitors"],
                ai_readiness=float(content["ai_readiness"]),
                trends=content["trends"]
            )
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}, Response: {content}")
            raise
        except Exception as e:
            self.logger.error(f"Analysis parsing error: {str(e)}")
            raise

    def _store_analysis(self, company: str, analysis: CompanyAnalysis) -> None:
        """Store analysis results in ChromaDB."""
        try:
            self.collection.upsert(
                ids=[company],
                documents=[json.dumps(analysis.to_dict())],
                metadatas=[{
                    "company": company,
                    "timestamp": str(analysis.timestamp),
                    "industry": analysis.industry
                }]
            )
        except Exception as e:
            self.logger.error(f"Storage error: {str(e)}")


    async def research_company(self, company: str) -> Dict:
        try:
        # Execute concurrent searches
            search_results = await asyncio.gather(
                self._search_company_info(company),
                self._search_industry_trends(company),
                self._search_competitors(company)
        )
        
        # Flatten and analyze results
            all_searches = [item for sublist in search_results for item in sublist]
            analysis = await self._analyze_findings(company, all_searches)
        
        # Store and return results
            self._store_analysis(company, analysis)
            return analysis.to_dict()
        
        except (asyncio.TimeoutError, ConnectionError) as e:
            self.logger.error(f"Network error researching {company}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error researching {company}: {str(e)}")
            raise
        