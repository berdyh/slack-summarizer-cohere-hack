"""
Evaluation Framework for Slack RAG Bot
Implements automated testing and quality assessment of RAG responses
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

# Import the main RAG service (will be imported from main.py)
# from main import answer_question, MESSAGE_VECTORS, INSTALLATIONS

# === Evaluation Data Models ===

@dataclass
class EvalQuestion:
    """Evaluation question with expected results"""
    query: str
    expected_sources: List[str]  # Message permalinks or IDs
    expected_answer_contains: List[str]  # Key phrases that should be in answer
    team_id: str
    channel_id: Optional[str] = None
    time_window_days: int = 30
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"  # general, technical, decision, planning
    expected_confidence_min: float = 0.3

@dataclass
class EvalResult:
    """Result of evaluating a single question"""
    question: EvalQuestion
    actual_answer: str
    actual_sources: List[str]
    confidence_score: float
    response_time: float
    source_recall: float  # Fraction of expected sources found
    answer_relevance: float  # Fraction of expected keywords found
    overall_score: float  # Combined score (0-1)
    errors: List[str] = None

@dataclass
class EvalReport:
    """Complete evaluation report"""
    team_id: str
    evaluation_time: datetime
    total_questions: int
    successful_questions: int
    average_response_time: float
    average_source_recall: float
    average_answer_relevance: float
    average_overall_score: float
    results: List[EvalResult]
    recommendations: List[str]

# === Gold Standard Evaluation Questions ===

# Sample evaluation questions for different scenarios
EVAL_QUESTIONS = [
    EvalQuestion(
        query="What did Sarah decide about the pricing strategy?",
        expected_sources=["message_1", "message_2"],
        expected_answer_contains=["pricing", "Sarah", "strategy", "decided"],
        team_id="T123",
        channel_id="C123",
        difficulty="medium",
        category="decision"
    ),
    EvalQuestion(
        query="Who is working on the mobile app features?",
        expected_sources=["message_3", "message_4"],
        expected_answer_contains=["mobile", "app", "features", "working"],
        team_id="T123",
        channel_id="C123",
        difficulty="easy",
        category="general"
    ),
    EvalQuestion(
        query="What are the main concerns about the new deployment?",
        expected_sources=["message_5", "message_6", "message_7"],
        expected_answer_contains=["deployment", "concerns", "issues", "problems"],
        team_id="T123",
        channel_id="C123",
        difficulty="hard",
        category="technical"
    ),
    EvalQuestion(
        query="When is the next team meeting scheduled?",
        expected_sources=["message_8"],
        expected_answer_contains=["meeting", "scheduled", "date", "time"],
        team_id="T123",
        channel_id="C123",
        difficulty="easy",
        category="planning"
    ),
    EvalQuestion(
        query="What was the outcome of the client feedback session?",
        expected_sources=["message_9", "message_10"],
        expected_answer_contains=["client", "feedback", "outcome", "session"],
        team_id="T123",
        channel_id="C123",
        difficulty="medium",
        category="decision"
    )
]

# === Evaluation Functions ===

class RAGEvaluator:
    """Main evaluation class for RAG system"""
    
    def __init__(self, rag_service_func):
        """
        Initialize evaluator with RAG service function
        
        Args:
            rag_service_func: Function that takes (team_id, query, **kwargs) and returns RAGResponse
        """
        self.rag_service = rag_service_func
        self.evaluation_history = []
    
    async def evaluate_question(self, question: EvalQuestion) -> EvalResult:
        """Evaluate a single question"""
        start_time = time.time()
        errors = []
        
        try:
            # Call the RAG service
            response = await self.rag_service(
                team_id=question.team_id,
                query=question.query,
                channel_id=question.channel_id,
                time_window_days=question.time_window_days
            )
            
            response_time = time.time() - start_time
            
            # Extract actual results
            actual_answer = response.answer
            actual_sources = [source.get('permalink', source.get('message_id', '')) for source in response.sources]
            confidence_score = response.confidence
            
            # Calculate metrics
            source_recall = self._calculate_source_recall(question.expected_sources, actual_sources)
            answer_relevance = self._calculate_answer_relevance(
                question.expected_answer_contains, 
                actual_answer
            )
            
            # Overall score (weighted combination)
            overall_score = (
                source_recall * 0.4 +  # 40% weight on source recall
                answer_relevance * 0.4 +  # 40% weight on answer relevance
                min(confidence_score, 1.0) * 0.2  # 20% weight on confidence
            )
            
            return EvalResult(
                question=question,
                actual_answer=actual_answer,
                actual_sources=actual_sources,
                confidence_score=confidence_score,
                response_time=response_time,
                source_recall=source_recall,
                answer_relevance=answer_relevance,
                overall_score=overall_score,
                errors=errors
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            errors.append(str(e))
            
            return EvalResult(
                question=question,
                actual_answer="",
                actual_sources=[],
                confidence_score=0.0,
                response_time=response_time,
                source_recall=0.0,
                answer_relevance=0.0,
                overall_score=0.0,
                errors=errors
            )
    
    async def evaluate_team(self, team_id: str, questions: List[EvalQuestion] = None) -> EvalReport:
        """Evaluate RAG performance for a specific team"""
        if questions is None:
            # Filter questions for this team
            questions = [q for q in EVAL_QUESTIONS if q.team_id == team_id]
        
        if not questions:
            raise ValueError(f"No evaluation questions found for team {team_id}")
        
        print(f"🔍 Starting evaluation for team {team_id} with {len(questions)} questions...")
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"  📝 Question {i}/{len(questions)}: {question.query[:50]}...")
            result = await self.evaluate_question(question)
            results.append(result)
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)
        
        # Calculate aggregate metrics
        successful_results = [r for r in results if not r.errors]
        total_questions = len(results)
        successful_questions = len(successful_results)
        
        if successful_results:
            average_response_time = np.mean([r.response_time for r in successful_results])
            average_source_recall = np.mean([r.source_recall for r in successful_results])
            average_answer_relevance = np.mean([r.answer_relevance for r in successful_results])
            average_overall_score = np.mean([r.overall_score for r in successful_results])
        else:
            average_response_time = 0.0
            average_source_recall = 0.0
            average_answer_relevance = 0.0
            average_overall_score = 0.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        
        report = EvalReport(
            team_id=team_id,
            evaluation_time=datetime.now(),
            total_questions=total_questions,
            successful_questions=successful_questions,
            average_response_time=average_response_time,
            average_source_recall=average_source_recall,
            average_answer_relevance=average_answer_relevance,
            average_overall_score=average_overall_score,
            results=results,
            recommendations=recommendations
        )
        
        # Store in history
        self.evaluation_history.append(report)
        
        return report
    
    def _calculate_source_recall(self, expected_sources: List[str], actual_sources: List[str]) -> float:
        """Calculate fraction of expected sources that were found"""
        if not expected_sources:
            return 1.0 if not actual_sources else 0.0
        
        # Simple string matching (in production, would use more sophisticated matching)
        found_sources = 0
        for expected in expected_sources:
            for actual in actual_sources:
                if expected.lower() in actual.lower() or actual.lower() in expected.lower():
                    found_sources += 1
                    break
        
        return found_sources / len(expected_sources)
    
    def _calculate_answer_relevance(self, expected_keywords: List[str], actual_answer: str) -> float:
        """Calculate fraction of expected keywords found in answer"""
        if not expected_keywords:
            return 1.0
        
        answer_lower = actual_answer.lower()
        found_keywords = 0
        
        for keyword in expected_keywords:
            if keyword.lower() in answer_lower:
                found_keywords += 1
        
        return found_keywords / len(expected_keywords)
    
    def _generate_recommendations(self, results: List[EvalResult]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Analyze response times
        response_times = [r.response_time for r in results if not r.errors]
        if response_times:
            avg_response_time = np.mean(response_times)
            if avg_response_time > 5.0:
                recommendations.append("Consider optimizing query processing - average response time is high")
            elif avg_response_time > 2.0:
                recommendations.append("Response times are acceptable but could be improved")
        
        # Analyze source recall
        source_recalls = [r.source_recall for r in results if not r.errors]
        if source_recalls:
            avg_source_recall = np.mean(source_recalls)
            if avg_source_recall < 0.5:
                recommendations.append("Source recall is low - consider improving retrieval algorithm")
            elif avg_source_recall < 0.7:
                recommendations.append("Source recall could be improved")
        
        # Analyze answer relevance
        answer_relevances = [r.answer_relevance for r in results if not r.errors]
        if answer_relevances:
            avg_answer_relevance = np.mean(answer_relevances)
            if avg_answer_relevance < 0.6:
                recommendations.append("Answer relevance is low - consider improving generation prompts")
            elif avg_answer_relevance < 0.8:
                recommendations.append("Answer relevance could be improved")
        
        # Analyze errors
        error_count = len([r for r in results if r.errors])
        if error_count > 0:
            recommendations.append(f"Address {error_count} evaluation errors to improve reliability")
        
        # Analyze confidence scores
        confidence_scores = [r.confidence_score for r in results if not r.errors]
        if confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            if avg_confidence < 0.3:
                recommendations.append("Low confidence scores - consider improving embedding quality or retrieval")
        
        if not recommendations:
            recommendations.append("System performance is good - continue monitoring")
        
        return recommendations
    
    def get_evaluation_summary(self, team_id: str = None) -> Dict[str, Any]:
        """Get summary of evaluation history"""
        if team_id:
            reports = [r for r in self.evaluation_history if r.team_id == team_id]
        else:
            reports = self.evaluation_history
        
        if not reports:
            return {"message": "No evaluations found"}
        
        latest_report = reports[-1]
        
        return {
            "team_id": team_id or "all",
            "total_evaluations": len(reports),
            "latest_evaluation": latest_report.evaluation_time.isoformat(),
            "latest_metrics": {
                "total_questions": latest_report.total_questions,
                "successful_questions": latest_report.successful_questions,
                "success_rate": latest_report.successful_questions / latest_report.total_questions,
                "average_response_time": latest_report.average_response_time,
                "average_source_recall": latest_report.average_source_recall,
                "average_answer_relevance": latest_report.average_answer_relevance,
                "average_overall_score": latest_report.average_overall_score
            },
            "recommendations": latest_report.recommendations
        }

# === Automated Evaluation Functions ===

async def run_automated_evaluation(team_id: str, rag_service_func) -> EvalReport:
    """Run automated evaluation for a team"""
    evaluator = RAGEvaluator(rag_service_func)
    return await evaluator.evaluate_team(team_id)

async def run_benchmark_evaluation(rag_service_func, teams: List[str] = None) -> Dict[str, EvalReport]:
    """Run evaluation across multiple teams"""
    if teams is None:
        # Get teams from installations (would need to import from main.py)
        teams = ["T123"]  # Placeholder
    
    results = {}
    evaluator = RAGEvaluator(rag_service_func)
    
    for team_id in teams:
        try:
            print(f"🚀 Running evaluation for team {team_id}...")
            report = await evaluator.evaluate_team(team_id)
            results[team_id] = report
            print(f"✅ Completed evaluation for team {team_id}")
        except Exception as e:
            print(f"❌ Failed evaluation for team {team_id}: {e}")
            results[team_id] = None
    
    return results

# === Evaluation Report Generation ===

def generate_evaluation_report(report: EvalReport) -> str:
    """Generate a human-readable evaluation report"""
    report_text = f"""
# RAG System Evaluation Report

**Team ID:** {report.team_id}
**Evaluation Time:** {report.evaluation_time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Questions:** {report.total_questions}
- **Successful Questions:** {report.successful_questions}
- **Success Rate:** {(report.successful_questions / report.total_questions * 100):.1f}%
- **Average Response Time:** {report.average_response_time:.2f}s
- **Average Source Recall:** {report.average_source_recall:.2f}
- **Average Answer Relevance:** {report.average_answer_relevance:.2f}
- **Overall Score:** {report.average_overall_score:.2f}

## Recommendations
"""
    
    for i, rec in enumerate(report.recommendations, 1):
        report_text += f"{i}. {rec}\n"
    
    report_text += "\n## Detailed Results\n"
    
    for i, result in enumerate(report.results, 1):
        report_text += f"""
### Question {i}: {result.question.query}
- **Expected Sources:** {len(result.question.expected_sources)}
- **Found Sources:** {len(result.actual_sources)}
- **Source Recall:** {result.source_recall:.2f}
- **Answer Relevance:** {result.answer_relevance:.2f}
- **Confidence:** {result.confidence_score:.2f}
- **Response Time:** {result.response_time:.2f}s
- **Overall Score:** {result.overall_score:.2f}
"""
        
        if result.errors:
            report_text += f"- **Errors:** {', '.join(result.errors)}\n"
    
    return report_text

# === Continuous Evaluation ===

class ContinuousEvaluator:
    """Continuous evaluation system for monitoring RAG performance"""
    
    def __init__(self, rag_service_func, evaluation_interval_hours: int = 24):
        self.rag_service = rag_service_func
        self.evaluation_interval = evaluation_interval_hours * 3600  # Convert to seconds
        self.last_evaluation = {}
        self.evaluator = RAGEvaluator(rag_service_func)
    
    async def should_evaluate_team(self, team_id: str) -> bool:
        """Check if team should be evaluated based on interval"""
        if team_id not in self.last_evaluation:
            return True
        
        time_since_last = time.time() - self.last_evaluation[team_id]
        return time_since_last >= self.evaluation_interval
    
    async def evaluate_if_needed(self, team_id: str) -> Optional[EvalReport]:
        """Evaluate team if evaluation is needed"""
        if await self.should_evaluate_team(team_id):
            print(f"🔄 Running scheduled evaluation for team {team_id}")
            report = await self.evaluator.evaluate_team(team_id)
            self.last_evaluation[team_id] = time.time()
            return report
        return None
    
    async def start_continuous_evaluation(self, teams: List[str]):
        """Start continuous evaluation loop"""
        while True:
            for team_id in teams:
                try:
                    await self.evaluate_if_needed(team_id)
                except Exception as e:
                    print(f"❌ Continuous evaluation error for team {team_id}: {e}")
            
            # Wait before next evaluation cycle
            await asyncio.sleep(3600)  # Check every hour
