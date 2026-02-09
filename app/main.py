import argparse
import os
import subprocess
import sys
import json
from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

TOOLS = [
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
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "required": ["file_path", "content"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to write to",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute",
                    }
                },
            },
        },
    },
]


def call_llm(client, messages):
    chat = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=messages,
        tools=TOOLS,
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

    while True:
        # Detect tool calls in response
        chat = call_llm(client, messages)
        msg = chat.choices[0].message
        # Append AI response
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": msg.tool_calls,
            }
        )

        if not msg.tool_calls:
            final = msg.content or ""
            break

        for tool_call in msg.tool_calls:
            type = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            match type:
                case "Read":
                    # Get path and read file contents
                    path = args["file_path"]

                    if os.path.exists(path):
                        with open(path, "r", encoding="utf-8") as f:
                            result = f.read()

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )
                case "Write":
                    # Get path and content
                    path = args["file_path"]
                    content = args["content"]

                    with open(path, "w", encoding="utf-8") as f:
                        result = f.write(content)

                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": "OK"}
                    )
                case "Bash":
                    cmd = args["command"]
                    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
                    out = (res.stdout or "") + (res.stderr or "")
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": out}
                    )


    print(chat.choices[0].message.content)


if __name__ == "__main__":
    main()
