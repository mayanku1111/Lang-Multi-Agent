from langchain_openai import ChatOpenAI
from typing import Dict, List, Optional
import asyncio
import aiohttp
import logging
from langchain_community.tools import TavilySearchResults
from . import UseCase, Resource

class EnhancedResourceCollector:
    def __init__(self, config: Dict):
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize APIs with proper error handling
        try:
            self.search_tool = TavilySearchResults(api_key=config.get("tavily_api_key"))
        except Exception as e:
            self.logger.error(f"Failed to initialize Tavily search: {str(e)}")
            raise

        # Store API configurations
        self.github_token = config.get("github_api_token")
        self.kaggle_username = config.get("kaggle_username")
        self.kaggle_key = config.get("kaggle_key")
        
        # Initialize session
        self.session = None

    async def _init_session(self):
        """Initialize aiohttp session if not already created"""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def _cleanup_session(self):
        """Cleanup aiohttp session"""
        if self.session is not None:
            await self.session.close()
            self.session = None

    async def collect_resources(self, use_cases: List[UseCase]) -> Dict[str, List[Resource]]:
        """Collect resources for given use cases with proper error handling"""
        resources = {}
        
        try:
            await self._init_session()
            
            for use_case in use_cases:
                try:
                    # Gather resources with individual error handling
                    gathered_resources = await asyncio.gather(
                        self._search_datasets(use_case),
                        self._search_github(use_case),
                        self._search_documentation(use_case),
                        return_exceptions=True
                    )
                    
                    # Filter out errors and flatten valid results
                    valid_resources = []
                    for result in gathered_resources:
                        if isinstance(result, Exception):
                            self.logger.error(f"Error collecting resources for {use_case.title}: {str(result)}")
                        elif isinstance(result, list):
                            valid_resources.extend(result)
                    
                    resources[use_case.title] = valid_resources

                except Exception as e:
                    self.logger.error(f"Error processing use case {use_case.title}: {str(e)}")
                    resources[use_case.title] = []
                    
        except Exception as e:
            self.logger.error(f"Critical error in collect_resources: {str(e)}")
            raise
        finally:
            await self._cleanup_session()
            
        return resources

    async def _search_datasets(self, use_case: UseCase) -> List[Resource]:
        """Search for relevant datasets"""
        try:
            if not self.kaggle_username or not self.kaggle_key:
                self.logger.warning("Kaggle credentials not provided")
                return []

            # Implement dataset search logic here
            # This is a placeholder implementation
            query = f"{use_case.title} {use_case.description} dataset"
            results = await self.search_tool.ainvoke({"query": query})
            
            return [
                Resource(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    type="dataset",
                    description=result.get("snippet", "")
                )
                for result in results
                if await self._validate_resource(Resource(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    type="dataset",
                    description=result.get("snippet", "")
                ))
            ]
            
        except Exception as e:
            self.logger.error(f"Error in _search_datasets: {str(e)}")
            return []

    async def _search_github(self, use_case: UseCase) -> List[Resource]:
        """Search for relevant GitHub repositories"""
        try:
            if not self.github_token:
                self.logger.warning("GitHub token not provided")
                return []

            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }

            query = f"{use_case.title} {use_case.description}"
            async with self.session.get(
                f"https://api.github.com/search/repositories?q={query}",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        Resource(
                            title=repo["name"],
                            url=repo["html_url"],
                            type="github_repo",
                            description=repo.get("description", "")
                        )
                        for repo in data.get("items", [])[:5]  # Limit to top 5 results
                    ]
                else:
                    self.logger.error(f"GitHub API error: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error in _search_github: {str(e)}")
            return []

    async def _search_documentation(self, use_case: UseCase) -> List[Resource]:
        """Search for relevant documentation"""
        try:
            query = f"{use_case.title} {use_case.description} documentation tutorial"
            results = await self.search_tool.ainvoke({"query": query})
            
            return [
                Resource(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    type="documentation",
                    description=result.get("snippet", "")
                )
                for result in results
                if await self._validate_resource(Resource(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    type="documentation",
                    description=result.get("snippet", "")
                ))
            ]
            
        except Exception as e:
            self.logger.error(f"Error in _search_documentation: {str(e)}")
            return []

    async def _validate_resource(self, resource: Resource) -> bool:
        """Validate if a resource URL is accessible"""
        try:
            async with self.session.head(resource.url, allow_redirects=True) as response:
                return response.status == 200
        except Exception as e:
            self.logger.debug(f"Resource validation failed for {resource.url}: {str(e)}")
            return False