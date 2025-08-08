import os
from google import genai
from dotenv import load_dotenv
from google.genai import types
import argparse


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="AI Agent using Gemini API")
    parser.add_argument("prompt", help="The prompt to send to the AI")
    parser.add_argument("--model", default="gemini-2.0-flash-001", help="Model to use (default: gemini-2.0-flash-001)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show token usage information")
    
    args = parser.parse_args()
    
    messages = [
        types.Content(role="user", parts=[types.Part(text=args.prompt)])
    ]
    
    response = client.models.generate_content(
        model=args.model,
        contents=messages,
    )
    
    if args.verbose:
        print("User prompt:", args.prompt)
        print("Prompt tokens:", response.usage_metadata.prompt_token_count)
        print("Response tokens:", response.usage_metadata.candidates_token_count)
        
    
    print(response.text)


if __name__ == "__main__":
    main()  
