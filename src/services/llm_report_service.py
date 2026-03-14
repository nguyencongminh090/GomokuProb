import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from src.core.config import AnalysisConfig
from src.core.v2_worker import V2GameResult

class LLMReportService:
    """Service to handle communication with OpenAI-compatible LLM endpoints."""
    
    SYSTEM_PROMPT = """
You are an expert Gomoku anti-cheat analyst evaluating statistical data to declare a verdict on a player.
Your goal is to translate complex probabilities and mathematical metrics into a clear, compelling, punchy, and highly readable report.

You will be provided with a JSON object containing the game's analysis metrics.
Core Metrics:
- 'classification': The mathematical system's final verdict (e.g., Human, Suspicious, Cheater).
- 'p_cheat': The Bayesian posterior probability (0.0 to 1.0).
- 'lambda_mle': The exponential rate parameter of their winrate loss. (>20 is suspicious).
- 'mean_delta': The average winrate loss per move. Near 0 is suspicious.
- 'near_optimal_pct': Percentage of moves where winrate loss was < 2%.
- 'log_lr': Log Likelihood Ratio comparing the cheater model to the human model. (>5 indicates cheating).

Advanced Metrics (V3/V4):
- 'temperature_mle': The softmax temperature parameter. Lower values (near 0.1) mean perfect correlation with the engine's top choices.
- 'best_distribution_fit': The shape of their error curve (e.g., exponential, gamma, weibull).
- 'em_pi': The percentage of their moves that belong to the "Good Player" distribution vs the "Blunder" distribution.
- 'em_lambda_good': The lambda of their "Good" moves.
- 'em_lambda_blunder': The lambda of their "Blunder" moves.

Context:
- 'cusum_triggered': A Boolean indicating a sudden shift in play strength mid-game (switching a bot on/off).
- 'profile_history': Details from their historical profile.

INSTRUCTIONS:
1. Do not use generic essay paragraphs. Use a sharp, modern, bulleted format.
2. Structure the report to build up the evidence BEFORE revealing the final conclusion.
3. Section 1 (Analysis Breakdown): Break down 3-4 compelling metrics (including at least one advanced V3/V4 metric if relevant) to paint a picture of how the player played. Use bold bullet points.
4. Section 2 (Context & History): Briefly mention the CUSUM trigger (did they toggle a bot?) and their Profile History.
5. Section 3 (Final Verdict): Conclude with the exact Classification and a 1-sentence bottom-line summary of WHY.
6. Section 4 (Referee's Advice): Synthesize the data and provide actionable advice to a human referee. If there are contradictions (e.g., strong cheating metrics like low Temperature MLE and high Lambda, but an overall "Human" classification), explicitly call out these contradictions and advise the referee on whether manual review is warranted.
7. Emphasize that this is a statistical probability analysis, not absolute proof.
8. DO NOT hallucinate numbers. Use only the provided JSON data.
"""

    @staticmethod
    def generate_prompt_payload(result: V2GameResult) -> str:
        """Serializes the V2GameResult into a concise JSON payload for the prompt."""
        
        # Format profile data if present
        profile_data = "No sufficient history available."
        if result.profile_result and result.profile_result.status != "INSUFFICIENT_DATA":
            profile_data = (f"Status: {result.profile_result.status}, "
                            f"Z-Score/T-Stat: {result.profile_result.test_statistic:.2f}, "
                            f"Historical Lambda: {result.profile_result.lambda_bar:.2f}")

        # Determine Layer 2 CUSUM flag
        cusum_triggered = False
        if result.layer2_result and result.layer2_result.cusum_triggered:
            cusum_triggered = True

        payload_dict = {
            "classification": result.classification.classification,
            "p_cheat": f"{result.classification.p_cheat * 100:.1f}%",
            "lambda_mle": round(result.features.lambda_mle, 2) if result.features else 0.0,
            "mean_delta": f"{result.features.mean_delta * 100:.2f}%" if result.features else "0.0%",
            "near_optimal_pct": f"{result.features.near_optimal_ratio * 100:.1f}%" if result.features else "0.0%",
            "log_lr": round(result.log_likelihood_ratio, 2),
            "temperature_mle": round(result.temperature_mle, 3),
            "best_distribution_fit": result.best_distribution,
            "em_pi": f"{result.em_pi * 100:.1f}%",
            "em_lambda_good": round(result.em_lambda_good, 2),
            "em_lambda_blunder": round(result.em_lambda_blunder, 2),
            "cusum_triggered": cusum_triggered,
            "profile_history": profile_data
        }
        
        return json.dumps(payload_dict, indent=2)

    @staticmethod
    def call_llm_api(config: AnalysisConfig, prompt_payload: str) -> str:
        """Makes the blocking HTTP request to the LLM."""
        if not config.llm_api_key:
            raise ValueError("LLM API Key is missing. Please set it in Settings.")

        provider = config.llm_provider.strip().lower()
        full_prompt = f"{LLMReportService.SYSTEM_PROMPT}\n\nHere is the game data to analyze:\n\n{prompt_payload}"
        
        if provider == "gemini":
            # Native Gemini 1.5 REST API Format
            # Endpoint must be something like: https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=...
            # The config currently holds the openai compatibility endpoint. Let's dynamically construct the right one if it's the default.
            url = config.llm_endpoint_url
            if "openai" in url:
                # User left the default openAI compatible URL but chose gemini provider.
                # Let's fix the URL to the native one for them.
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.llm_model}:generateContent?key={config.llm_api_key}"
            else:
                # If they provided a custom URL, append the key if needed
                if "?" not in url:
                    url += f"?key={config.llm_api_key}"
            
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{"parts": [{"text": full_prompt}]}],
                "generationConfig": {"temperature": 0.3}
            }
            
            req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
            
            try:
                with urllib.request.urlopen(req, timeout=45) as response:
                    res_body = response.read().decode("utf-8")
                    res_json = json.loads(res_body)
                    
                    if "candidates" in res_json and len(res_json["candidates"]) > 0:
                        return res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                    else:
                        raise ValueError("Unexpected response format from Gemini API.")
                        
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8")
                raise Exception(f"HTTP Error {e.code}: {e.reason}\n{err_body}")
            except Exception as e:
                raise Exception(f"Error calling Gemini: {str(e)}")
                
        else:
            # Standard OpenAI Compatible Format (Groq, OpenAI, LMStudio)
            url = config.llm_endpoint_url
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.llm_api_key}"
            }
            data = {
                "model": config.llm_model,
                "messages": [
                    {"role": "system", "content": LLMReportService.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Here is the game data to analyze:\n\n{prompt_payload}"}
                ],
                "temperature": 0.3
            }
            req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
            
            try:
                with urllib.request.urlopen(req, timeout=45) as response:
                    res_body = response.read().decode("utf-8")
                    res_json = json.loads(res_body)
                    if "choices" in res_json and len(res_json["choices"]) > 0:
                        return res_json["choices"][0]["message"]["content"].strip()
                    else:
                        raise ValueError("Unexpected response format from OpenAI compat API.")
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8")
                raise Exception(f"HTTP Error {e.code}: {e.reason}\n{err_body}")
            except Exception as e:
                raise Exception(f"Error calling LLM: {str(e)}")


class LLMReportWorker(QObject):
    """
    Worker to run the blocking LLM API call in a QThread.
    Emits the final report string or an error message.
    """
    report_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: AnalysisConfig, result: V2GameResult):
        super().__init__()
        self.config = config
        self.result = result

    def run(self):
        try:
            payload = LLMReportService.generate_prompt_payload(self.result)
            report_text = LLMReportService.call_llm_api(self.config, payload)
            self.report_ready.emit(report_text)
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.finished.emit()
