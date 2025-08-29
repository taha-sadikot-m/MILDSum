"""Configuration settings for Legal Summarization workflow."""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
# Load environment variables (dotenv optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use os.getenv directly


@dataclass
class GeminiConfig:
    """Configuration for Gemini LLM."""
    api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    model_name: str = "gemini-pro"
    temperature: float = 0.1
    max_tokens: int = 4096
    top_p: float = 0.8
    top_k: int = 40


@dataclass
class WorkflowConfig:
    """Configuration for the legal summarization workflow."""
    max_retries: int = 3
    timeout_seconds: int = 300
    enable_parallel_processing: bool = True
    cache_enabled: bool = True
    log_level: str = "INFO"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    rouge_types: list = field(default_factory=lambda: ["rouge1", "rouge2", "rougeL"])
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"
    enable_bertscore: bool = True
    evaluation_batch_size: int = 16


@dataclass
class Settings:
    """Main configuration class."""
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Storage paths
    output_dir: str = "output"
    cache_dir: str = ".cache"
    logs_dir: str = "logs"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Only require API key if we're not in testing mode
        if not self.gemini.api_key and os.getenv("REQUIRE_API_KEY", "true").lower() == "true":
            raise ValueError("GOOGLE_API_KEY environment variable is required")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Settings":
        """Create Settings from dictionary."""
        return cls(
            gemini=GeminiConfig(**config_dict.get("gemini", {})),
            workflow=WorkflowConfig(**config_dict.get("workflow", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
            **{k: v for k, v in config_dict.items() 
               if k not in ["gemini", "workflow", "evaluation"]}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Settings to dictionary."""
        return {
            "gemini": {
                "api_key": "***" if self.gemini.api_key else "",
                "model_name": self.gemini.model_name,
                "temperature": self.gemini.temperature,
                "max_tokens": self.gemini.max_tokens,
                "top_p": self.gemini.top_p,
                "top_k": self.gemini.top_k,
            },
            "workflow": {
                "max_retries": self.workflow.max_retries,
                "timeout_seconds": self.workflow.timeout_seconds,
                "enable_parallel_processing": self.workflow.enable_parallel_processing,
                "cache_enabled": self.workflow.cache_enabled,
                "log_level": self.workflow.log_level,
            },
            "evaluation": {
                "rouge_types": self.evaluation.rouge_types,
                "bertscore_model": self.evaluation.bertscore_model,
                "enable_bertscore": self.evaluation.enable_bertscore,
                "evaluation_batch_size": self.evaluation.evaluation_batch_size,
            },
            "output_dir": self.output_dir,
            "cache_dir": self.cache_dir,
            "logs_dir": self.logs_dir,
        }


# Global settings instance
settings = Settings()