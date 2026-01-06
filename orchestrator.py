import os
import time
import uuid
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Agent:
    """
    Represents a single LLM agent in the simulation.
    """
    def __init__(self, model_slug: str, system_prompt: str, name: str):
        self.model_slug = model_slug
        self.system_prompt = system_prompt
        self.name = name
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Initialize OpenAI client for OpenRouter
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables.")
            
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def generate_response(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """
        Generates a response from the agent based on input text.
        Accepts optional kwargs for model parameters (temperature, top_p, etc.)
        """
        # Add user input to history
        self.history.append({"role": "user", "content": input_text})
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_slug,
                messages=self.history,
                **kwargs
            )

            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            # Check for refusal (empty content or specific flag if available)
            is_refusal = not content or finish_reason == "content_filter"
            
            # Add assistant response to history
            if content:
                self.history.append({"role": "assistant", "content": content})
            
            return {
                "content": content,
                "latency_ms": latency_ms,
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
                "finish_reason": finish_reason,
                "is_refusal": is_refusal
            }
            
        except Exception as e:
            logger.error(f"Error generating response for {self.name}: {e}")
            raise e

class Orchestrator:
    """
    Manages the simulation between two agents.
    """
    def __init__(self, 
                 agent_a_config: Dict[str, Any], 
                 agent_b_config: Dict[str, Any], 
                 scenario_name: str,
                 experiment_id: str = None):
        
        self.experiment_id = experiment_id or str(uuid.uuid4())
        self.scenario_name = scenario_name
        
        # Initialize agents with full config
        self.agent_a = Agent(
            model_slug=agent_a_config["model"],
            system_prompt=agent_a_config["system_prompt"],
            name="Agent A"
        )
        self.agent_b = Agent(
            model_slug=agent_b_config["model"],
            system_prompt=agent_b_config["system_prompt"],
            name="Agent B"
        )

        # Store user-facing persona snapshots for clean exports
        self.persona_a_snapshot = agent_a_config.get("user_persona_snapshot", self.agent_a.system_prompt)
        self.persona_b_snapshot = agent_b_config.get("user_persona_snapshot", self.agent_b.system_prompt)
        
        # Store advanced params if present, else default
        self.agent_a_params = agent_a_config.get("params", {})
        self.agent_b_params = agent_b_config.get("params", {})
        
        self.logs: List[Dict[str, Any]] = []

    def run_simulation(self, num_turns: int, initial_message: str = "Hello."):
        """
        Runs the conversation loop for a specified number of turns.
        Yields the log entry for each turn as it happens.
        """
        logger.info(f"Starting simulation {self.experiment_id} - Scenario: {self.scenario_name}")
        
        # Initial trigger for Agent A to start the conversation
        last_message = initial_message 
        
        for turn_id in range(num_turns):
            # --- Agent A Turn ---
            try:
                logger.info(f"Turn {turn_id}: Agent A generating response...")
                # Pass advanced params (temperature, etc.) if supported by Agent class
                # For now, we assume Agent.generate_response can handle them or we update Agent class too.
                # Let's update Agent class in a separate step or assume it uses defaults.
                # Ideally, we pass kwargs to generate_response.
                
                response_a = self.agent_a.generate_response(last_message, **self.agent_a_params)
                log_entry_a = self._create_log_entry(turn_id, self.agent_a, self.agent_b, response_a)
                self.logs.append(log_entry_a)
                yield log_entry_a
                
                last_message = response_a["content"]
                
                if response_a["is_refusal"]:
                    logger.warning("Agent A refused to respond. Ending simulation.")
                    break
            except Exception as e:
                logger.error(f"Failed turn {turn_id} for Agent A: {e}")
                break

            # --- Agent B Turn ---
            try:
                logger.info(f"Turn {turn_id}: Agent B generating response...")
                response_b = self.agent_b.generate_response(last_message, **self.agent_b_params)
                log_entry_b = self._create_log_entry(turn_id, self.agent_b, self.agent_a, response_b)
                self.logs.append(log_entry_b)
                yield log_entry_b
                
                last_message = response_b["content"]
                
                if response_b["is_refusal"]:
                    logger.warning("Agent B refused to respond. Ending simulation.")
                    break
            except Exception as e:
                logger.error(f"Failed turn {turn_id} for Agent B: {e}")
                break
                
        logger.info(f"Simulation {self.experiment_id} completed.")

    def _create_log_entry(self, turn_id: int, speaker: Agent, responder: Agent, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a log entry dictionary.
        """
        return {
            "experiment_id": self.experiment_id,
            "turn_id": turn_id,
            "scenario": self.scenario_name,
            "speaker_model": speaker.model_slug,
            "responder_model": responder.model_slug,
            "timestamp": datetime.utcnow().isoformat(),
            "latency_ms": response_data["latency_ms"],
            "input_tokens": response_data["input_tokens"],
            "output_tokens": response_data["output_tokens"],
            "content": response_data["content"],
            "finish_reason": response_data["finish_reason"],
            "is_refusal": response_data["is_refusal"],
            "system_prompt_snapshot": self.persona_a_snapshot if speaker.name == "Agent A" else self.persona_b_snapshot
        }

    def save_logs(self, filepath: str):
        """
        Saves the accumulated logs to a JSONL file.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'a', encoding='utf-8') as f:
            for entry in self.logs:
                f.write(json.dumps(entry) + '\n')
        
        logger.info(f"Logs saved to {filepath}")
