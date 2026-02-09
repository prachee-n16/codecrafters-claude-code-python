import argparse
import os
import sys
import json
from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://oxpenrouter.ai/api/v1")

def call_llm(client, messages):
    chat = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read and return the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to read",
                            }
                        },
                        "required": ["file_path"],
                    },
                },
            }
        ],
    )

    return chat

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    # store conversation history
    messages = [{"role": "user", "content": args.p}]
    
    chat = call_llm(client, messages)

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)
    
    # Detect tool calls in response
    msg = chat.choices[0].message
    # Append AI response
    messages.append(msg)
    while msg.tool_calls:
        for tool_call in msg.tool_calls:
            type = tool_call.function.name
            match type:
                case "Read":
                    # Parse arguments
                    args = json.loads(tool_call.function.arguments)
                    # Get path and read file contents
                    path = args["file_path"]

                    if os.path.exists(path):
                        f = open(path)
                        print(f.read())
                    continue
        
        chat = call_llm(client, messages)
        msg = chat.choices[0].message
        messages.append(msg)
            
    print(chat.choices[0].message.content) 

if __name__ == "__main__":
    main()
