
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client

def reproduce():
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    supabase = create_client(url, key)
    
    # Mocking exactly what Orchestrator._create_log_entry produces
    log_entry = {
        "experiment_id": "reproduction-test",
        "turn_id": 99,
        "scenario": "repro-scenario",
        "speaker_model": "gpt-repro",
        "responder_model": "gpt-repro",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": 123.456,
        "input_tokens": 10,
        "output_tokens": 20,
        "content": "Reproduction test content with special chars: 🚀",
        "finish_reason": "stop",
        "is_refusal": False,
        "system_prompt_snapshot": "Repro system prompt"
    }
    
    print(f"Attempting to insert log entry with timestamp: {log_entry['timestamp']}")
    try:
        response = supabase.table("session_logs").insert([log_entry]).execute()
        print(f"Success! Inserted row: {response.data}")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    reproduce()
