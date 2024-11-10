from typing import Dict, List
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from chromadb import PersistentClient

class ResearchAgent:
    def __init__(self):
        self.search_tool = TavilySearchResults()
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")
        self.db = PersistentClient(path="./chroma_db")
        self.collection = self.db.get_or_create_collection("market_research")
        
    def search_and_analyze(self, company: str) -> Dict:
        # Search for company information
        company_search = self.search_tool.invoke(
            f"company analysis {company} business model products services market focus"
        )
        
        # Search for industry trends
        industry_search = self.search_tool.invoke(
            f"{company} industry AI ML trends digital transformation"
        )
        
        # Analyze findings with LLM
        analysis_prompt = f"""
        Analyze the following information about {company}:
        
        Company Information:
        {company_search}
        
        Industry Trends:
        {industry_search}
        
        Provide a structured analysis including:
        1. Company Overview
        2. Key Products/Services
        3. Market Position
        4. Industry AI/ML Trends
        5. Potential AI Use Cases
        """
        
        analysis = self.llm.invoke(analysis_prompt)
        
        # Store in ChromaDB
        self.collection.add(
            documents=[str(analysis.content)],
            metadatas=[{"company": company, "type": "analysis"}],
            ids=[f"{company}_analysis"]
        )
        
        return {
            "company": company,
            "analysis": analysis.content,
            "sources": company_search + industry_search
        }
