from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import List, Dict, Optional

@dataclass
class CompanyAnalysis:
    company_name: str
    industry: str
    business_model: str
    key_products: List[str]
    market_position: str
    competitors: List[str]
    ai_readiness: float
    timestamp: datetime

@dataclass
class UseCase:
    title: str
    description: str
    business_impact: str
    complexity: str
    timeline: str
    priority_score: float
    required_resources: List[str]
    implementation_steps: List[str]

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