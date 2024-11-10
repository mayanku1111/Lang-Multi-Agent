# agents/__init__.py

from dataclasses import dataclass, field
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
    timestamp: datetime = field(default_factory=datetime.now)
    trends: str = ""

@dataclass
class UseCase:
    title: str
    description: str
    impact: str
    complexity: str
    timeline: str
    priority_score: float = 0.0
    required_resources: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    implementation_steps: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)

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