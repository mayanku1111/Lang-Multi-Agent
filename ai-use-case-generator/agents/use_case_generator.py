
import asyncio
import logging
import json
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from chromadb import PersistentClient
from . import CompanyAnalysis, UseCase

class EnhancedUseCaseGenerator:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.llm = ChatOpenAI(
            model=config.get("model_name", "gpt-4"),
            api_key=config["openai_api_key"],
            temperature=0.7
        )
        self.db = PersistentClient(path=config["chroma_path"])
        self.collection = self.db.get_or_create_collection("use_cases")
        
        self.use_case_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI solution architect specializing in identifying valuable AI use cases for businesses.
            Generate specific, actionable AI use cases based on the company analysis provided."""),
            ("human", """
            Company Analysis:
            Company: {company_name}
            Industry: {industry}
            Business Model: {business_model}
            Market Position: {market_position}
            AI Readiness: {ai_readiness}
            
            Generate 5 specific AI use cases that could benefit this company.
            Each use case must be a JSON object with exactly these keys:
            - title: string
            - description: string
            - impact: string
            - complexity: integer (1-5)
            - timeline: string
            - data_sources: array of strings
            - challenges: array of strings
            
            Respond with only the JSON array. No other text or formatting.
            """)
        ])

    def _clean_json_response(self, response: str) -> str:
        cleaned = response.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned[3:-3].strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        return cleaned

    def _use_case_to_dict(self, use_case: UseCase) -> Dict:
        return {
            'title': use_case.title,
            'description': use_case.description,
            'impact': use_case.impact,
            'complexity': use_case.complexity,
            'timeline': use_case.timeline,
            'priority_score': use_case.priority_score,
            'data_sources': use_case.data_sources,
            'challenges': use_case.challenges
        }

    async def _generate_initial_use_cases(self, analysis: CompanyAnalysis) -> List[UseCase]:
        try:
            chain = self.use_case_prompt | self.llm
            response = await chain.ainvoke({
                "company_name": analysis.company_name,
                "industry": analysis.industry,
                "business_model": analysis.business_model,
                "market_position": analysis.market_position,
                "ai_readiness": analysis.ai_readiness
            })
            
            cleaned_response = self._clean_json_response(response.content)
            use_case_data = json.loads(cleaned_response)
            
            if not isinstance(use_case_data, list):
                raise ValueError("Expected JSON array of use cases")
            
            use_cases = []
            for case in use_case_data:
                use_case = UseCase(
                    title=case["title"],
                    description=case["description"],
                    impact=case["impact"],
                    complexity=int(case["complexity"]),
                    timeline=case["timeline"],
                    data_sources=case.get("data_sources", []),
                    challenges=case.get("challenges", [])
                )
                use_cases.append(use_case)
            
            return use_cases
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}, Response: {response.content}")
            raise
        except Exception as e:
            self.logger.error(f"Error in initial use case generation: {str(e)}")
            raise

    async def _calculate_priority_score(self, use_case: UseCase, analysis: CompanyAnalysis) -> float:
        try:
            scoring_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an AI investment analyst. Score this use case's priority from 0-1."),
                ("human", f"""
                Company Analysis:
                - Industry: {analysis.industry}
                - AI Readiness: {analysis.ai_readiness}
                - Market Position: {analysis.market_position}
                
                Use Case:
                - Title: {use_case.title}
                - Description: {use_case.description}
                - Impact: {use_case.impact}
                - Complexity: {use_case.complexity}
                
                Consider:
                1. Alignment with company's market position
                2. Technical feasibility given AI readiness
                3. Expected ROI
                4. Implementation complexity
                
                Return only a number between 0 and 1.
                """)
            ])
            
            chain = scoring_prompt | self.llm
            response = await chain.ainvoke({})
            
            try:
                score = float(response.content.strip())
                return max(0.0, min(1.0, score))
            except ValueError:
                self.logger.error(f"Invalid score format received: {response.content}")
                return 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating priority score: {str(e)}")
            return 0.5

    async def _score_use_cases(self, use_cases: List[UseCase], analysis: CompanyAnalysis) -> List[UseCase]:
        try:
            for use_case in use_cases:
                use_case.priority_score = await self._calculate_priority_score(use_case, analysis)
            return sorted(use_cases, key=lambda x: x.priority_score, reverse=True)
        except Exception as e:
            self.logger.error(f"Error scoring use cases: {str(e)}")
            raise

    def _store_use_cases(self, company_name: str, use_cases: List[UseCase]) -> None:
        try:
            docs = []
            metadatas = []
            ids = []
            
            for i, use_case in enumerate(use_cases):
                doc_id = f"{company_name}_usecase_{i}"
                docs.append(json.dumps(self._use_case_to_dict(use_case)))
                metadatas.append({
                    "company": company_name,
                    "title": use_case.title,
                    "priority_score": use_case.priority_score,
                    "timestamp": str(datetime.now())
                })
                ids.append(doc_id)
            
            self.collection.upsert(
                ids=ids,
                documents=docs,
                metadatas=metadatas
            )
            
        except Exception as e:
            self.logger.error(f"Error storing use cases: {str(e)}")

    async def generate_use_cases(self, analysis_data: Dict) -> List[Dict]:
        try:
            analysis = CompanyAnalysis.from_dict(analysis_data)
            use_cases = await self._generate_initial_use_cases(analysis)
            scored_use_cases = await self._score_use_cases(use_cases, analysis)
            self._store_use_cases(analysis.company_name, scored_use_cases)
            return [self._use_case_to_dict(uc) for uc in scored_use_cases]
        except Exception as e:
            self.logger.error(f"Error generating use cases: {str(e)}")
            raise

    async def _search_datasets(self, use_case: UseCase) -> List[Dict]:
        datasets = []
        search_query = f"dataset {use_case.title} machine learning AI"
    
    # Search Kaggle datasets
        if self.config["kaggle_api"]:
            try:
                kaggle_results = await self._search_kaggle(search_query)
                datasets.extend(kaggle_results)
            except Exception as e:
                self.logger.error(f"Kaggle search error: {str(e)}")
    
    # Search Hugging Face datasets
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://huggingface.co/api/datasets?search={search_query}"
                async with session.get(url) as response:
                    if response.status == 200:
                        results = await response.json()
                        for result in results[:5]:
                            datasets.append({
                            'type': ResourceType.DATASET,
                            'title': result['name'],
                            'url': f"https://huggingface.co/datasets/{result['id']}",
                            'description': result.get('description', ''),
                            'relevance_score': 0.8,
                            'use_case_id': use_case.title
                        })
        except Exception as e:
            self.logger.error(f"HuggingFace search error: {str(e)}")
    
        return datasets 