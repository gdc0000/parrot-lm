
import os
from dotenv import load_dotenv
from supabase import create_client

def check_schema():
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    supabase = create_client(url, key)
    
    print("Fetching one row from session_logs...")
    try:
        response = supabase.table("session_logs").select("*").limit(1).execute()
        if response.data:
            print("Row found! Columns:")
            for k, v in response.data[0].items():
                print(f"  - {k}: {type(v).__name__} (example: {v})")
        else:
            print("No rows found in session_logs.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_schema()
