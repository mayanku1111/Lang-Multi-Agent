from typing import Dict, List, Optional
from langchain_community.tools import TavilySearchResults
import asyncio
import aiohttp
import logging
from dataclasses import dataclass
from enum import Enum

@dataclass
class UseCase:
    title: str
    description: str
    impact: str
    complexity: str
    timeline: str

class ResourceType(Enum):
    DATASET = "dataset"
    DOCUMENTATION = "documentation" 
    GITHUB_REPO = "github_repo"
    RESEARCH_PAPER = "research_paper"

@dataclass
class Resource:
    type: ResourceType
    title: str
    url: str
    description: str
    relevance_score: float
    use_case_id: str

class EnhancedResourceCollector:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.search_tool = TavilySearchResults(api_key=config["tavily_api_key"])
        self.github_token = config.get("github_api")
        self.session: Optional[aiohttp.ClientSession] = None
        self.max_retries = 3
        self.retry_delay = 1

    async def collect_resources(self, use_cases: List[Dict]) -> Dict[str, Dict]:
        """Collect resources for each use case."""
        resources = {}
        try:
            for use_case in use_cases:
                use_case_obj = UseCase(
                    title=use_case["title"],
                    description=use_case["description"],
                    impact=use_case["impact"],
                    complexity=use_case["complexity"],
                    timeline=use_case["timeline"]
                )
                resources[use_case_obj.title] = {
                    "datasets": await self._search_datasets(use_case_obj),
                    "documentation": await self._search_documentation(use_case_obj),
                    "github": await self._search_github(use_case_obj)
                }
        except Exception as e:
            self.logger.error(f"Error collecting resources: {str(e)}")
        finally:
            if self.session and not self.session.closed:
                await self.session.close()
        return resources

    async def _search_datasets(self, use_case: UseCase) -> List[Dict]:
        """Search for relevant datasets."""
        datasets = []
        try:
            query = f"dataset machine learning {use_case.title} {use_case.description}"
            results = await self.search_tool.ainvoke({"query": query})
            
            for result in results:
                if any(term in result["url"].lower() for term in ["kaggle.com", "huggingface.co", "data.gov", "github"]):
                    datasets.append({
                        "type": ResourceType.DATASET,
                        "title": result["title"],
                        "url": result["url"],
                        "description": result["snippet"],
                        "relevance_score": result.get("relevance_score", 0.0),
                        "use_case_id": use_case.title
                    })
            
            return datasets[:5]  # Return top 5 most relevant datasets
            
        except Exception as e:
            self.logger.error(f"Dataset search error for {use_case.title}: {str(e)}")
            return []

    async def _search_documentation(self, use_case: UseCase) -> List[Dict]:
        """Search for relevant documentation and tutorials."""
        try:
            query = f"tutorial documentation guide {use_case.title} machine learning AI implementation"
            results = await self.search_tool.ainvoke({"query": query})
            
            docs = []
            for result in results:
                if any(term in result["url"].lower() for term in [
                    "docs.", "documentation", "guide", "tutorial", "blog.", "medium.com"
                ]):
                    docs.append({
                        "type": ResourceType.DOCUMENTATION,
                        "title": result["title"],
                        "url": result["url"],
                        "description": result["snippet"],
                        "relevance_score": result.get("relevance_score", 0.0),
                        "use_case_id": use_case.title
                    })
            
            return docs[:5]
            
        except Exception as e:
            self.logger.error(f"Documentation search error for {use_case.title}: {str(e)}")
            return []

    async def _search_github(self, use_case: UseCase) -> List[Dict]:
        """Search for relevant GitHub repositories."""
        try:
            if not self.github_token:
                return []
                
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            query = f"{use_case.title} machine learning"
            url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    repos = []
                    
                    for repo in data["items"][:5]:
                        repos.append({
                            "type": ResourceType.GITHUB_REPO,
                            "title": repo["full_name"],
                            "url": repo["html_url"],
                            "description": repo["description"] or "",
                            "relevance_score": 0.8,
                            "use_case_id": use_case.title
                        })
                        
                    return repos
                    
            return []
            
        except Exception as e:
            self.logger.error(f"GitHub search error for {use_case.title}: {str(e)}")
            return []