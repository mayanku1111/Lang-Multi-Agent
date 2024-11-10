import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from chromadb import PersistentClient
from . import CompanyAnalysis, UseCase

class EnhancedUseCaseGenerator:
    def __init__(self, config: Dict):
        """Initialize the use case generator with configuration."""
        self.logger = logging.getLogger(__name__)
        self.llm = ChatOpenAI(
            model=config.get("model_name", "gpt-4"),
            api_key=config["openai_api_key"],
            temperature=0.7
        )
        self.db = PersistentClient(path=config["chroma_path"])
        self.collection = self.db.get_or_create_collection("use_cases")
        
        # Create use case generation prompt template
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
            Each use case should include:
            1. Title
            2. Description
            3. Expected Impact
            4. Implementation Complexity (1-5)
            5. Required Data Sources
            6. Key Challenges

            Format the response as a list of JSON objects.
            """)
        ])

    async def generate_use_cases(self, analysis: CompanyAnalysis) -> List[UseCase]:
        """Generate and process AI use cases for a company."""
        try:
            initial_use_cases = await self._generate_initial_use_cases(analysis)
            scored_use_cases = await self._score_use_cases(initial_use_cases, analysis)
            self._store_use_cases(analysis.company_name, scored_use_cases)
            return scored_use_cases
        except Exception as e:
            self.logger.error(f"Error generating use cases: {str(e)}")
            raise

    async def _generate_initial_use_cases(self, analysis: CompanyAnalysis) -> List[UseCase]:
        """Generate initial use cases using LLM."""
        try:
            # Prepare the prompt with company analysis
            chain = self.use_case_prompt | self.llm
            
            response = await chain.ainvoke({
                "company_name": analysis.company_name,
                "industry": analysis.industry,
                "business_model": analysis.business_model,
                "market_position": analysis.market_position,
                "ai_readiness": analysis.ai_readiness
            })
            
            # Parse response into UseCase objects
            use_cases = []
            for case in eval(response.content):
                use_case = UseCase(
                    title=case["Title"],
                    description=case["Description"],
                    expected_impact=case["Expected Impact"],
                    complexity=case["Implementation Complexity"],
                    data_sources=case["Required Data Sources"],
                    challenges=case["Key Challenges"],
                    priority_score=0.0  # Will be calculated later
                )
                use_cases.append(use_case)
                
            return use_cases
            
        except Exception as e:
            self.logger.error(f"Error in initial use case generation: {str(e)}")
            raise

    async def _calculate_priority_score(self, use_case: UseCase, analysis: CompanyAnalysis) -> float:
        """Calculate priority score for a use case based on multiple factors."""
        try:
            # Create scoring prompt
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
                - Expected Impact: {use_case.expected_impact}
                - Complexity: {use_case.complexity}

                Consider:
                1. Alignment with company's market position
                2. Technical feasibility given AI readiness
                3. Expected ROI
                4. Implementation complexity
                
                Return only a float number between 0 and 1.
                """)
            ])

            # Get score from LLM
            chain = scoring_prompt | self.llm
            response = await chain.ainvoke({})
            
            # Parse and validate score
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))  # Ensure score is between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating priority score: {str(e)}")
            return 0.5  # Default score if calculation fails

    async def _score_use_cases(self, use_cases: List[UseCase], analysis: CompanyAnalysis) -> List[UseCase]:
        """Score and sort use cases by priority."""
        try:
            scoring_tasks = []
            for use_case in use_cases:
                task = self._calculate_priority_score(use_case, analysis)
                scoring_tasks.append(task)
                
            # Calculate scores concurrently
            scores = await asyncio.gather(*scoring_tasks)
            
            # Assign scores to use cases
            for use_case, score in zip(use_cases, scores):
                use_case.priority_score = score
                
            # Sort by priority score
            return sorted(use_cases, key=lambda x: x.priority_score, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error scoring use cases: {str(e)}")
            return use_cases

    def _store_use_cases(self, company_name: str, use_cases: List[UseCase]) -> None:
        """Store generated use cases in ChromaDB."""
        try:
            # Prepare documents and metadata for storage
            docs = []
            metadatas = []
            ids = []
            
            for i, use_case in enumerate(use_cases):
                doc_id = f"{company_name}_usecase_{i}"
                docs.append(str(use_case.__dict__))
                metadatas.append({
                    "company": company_name,
                    "title": use_case.title,
                    "priority_score": use_case.priority_score,
                    "timestamp": str(datetime.now())
                })
                ids.append(doc_id)
            
            # Store in ChromaDB
            self.collection.upsert(
                ids=ids,
                documents=docs,
                metadatas=metadatas
            )
            
        except Exception as e:
            self.logger.error(f"Error storing use cases: {str(e)}")