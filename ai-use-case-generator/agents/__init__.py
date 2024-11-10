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

    def to_dict(self) -> Dict:
        """Convert CompanyAnalysis to dictionary."""
        return {
            "company_name": self.company_name,
            "industry": self.industry,
            "business_model": self.business_model,
            "key_products": self.key_products,
            "market_position": self.market_position,
            "competitors": self.competitors,
            "ai_readiness": self.ai_readiness,
            "trends": self.trends,
            "timestamp": str(self.timestamp),
            "analysis": f"""Industry: {self.industry}
Business Model: {self.business_model}
Key Products: {', '.join(self.key_products)}
Market Position: {self.market_position}
Competitors: {', '.join(self.competitors)}
AI Readiness Score: {self.ai_readiness}"""
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CompanyAnalysis':
        """Create CompanyAnalysis from dictionary."""
        return cls(
            company_name=data["company_name"],
            industry=data["industry"],
            business_model=data["business_model"],
            key_products=data["key_products"],
            market_position=data["market_position"],
            competitors=data["competitors"],
            ai_readiness=float(data["ai_readiness"]),
            trends=data.get("trends", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now()
        )

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