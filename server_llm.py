import ctypes
import sys
import subprocess
import socket
from flask import Flask, request, jsonify
from flask_cors import CORS
import ollama
from ddgs import DDGS

# -----------------------------
# FIREWALL MANAGEMENT
# -----------------------------

RULE_NAME = "OllamaChat18888"
PORT = "18888"


def is_admin():
    """Check if running as admin."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def firewall_rule_exists():
    """Check if the firewall rule already exists."""
    result = subprocess.run(
        ["netsh", "advfirewall", "firewall", "show", "rule", f"name={RULE_NAME}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )
    return RULE_NAME.lower() in result.stdout.lower()


def create_firewall_rule():
    """Create the firewall rule."""
    print("Creating firewall rule...")
    subprocess.run(
        [
            "netsh", "advfirewall", "firewall", "add", "rule",
            f"name={RULE_NAME}",
            "dir=in",
            "action=allow",
            "protocol=TCP",
            f"localport={PORT}",
            "profile=private"
        ],
        shell=True
    )
    print("Firewall rule added.")


def ensure_firewall_rule():
    """Elevate if needed, then create rule if missing."""
    if not firewall_rule_exists():
        print("Firewall rule missing.")
        if not is_admin():
            print("Re-launching with admin rights...")
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, " ".join(sys.argv), None, 1
            )
            sys.exit()
        else:
            create_firewall_rule()
    else:
        print("Firewall rule already exists.")


# -----------------------------
# WEB SEARCH
# -----------------------------

SEARCH_TOOL_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the internet for up-to-date information. "
                "Use this when the user asks about recent events, current data, "
                "or anything you are not confident about."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to retrieve (default: 5).",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def search_web(query: str, max_results: int = 5) -> str:
    """Run a DuckDuckGo search and return formatted results as a string."""
    print(f"[search_web] query='{query}' max_results={max_results}")
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    if not results:
        return "No results found."

    formatted = "\n\n".join([
        f"Title: {r['title']}\nSummary: {r['body']}\nURL: {r['href']}"
        for r in results
    ])
    return formatted


def dispatch_tool_call(tool_call) -> str:
    """Execute a tool call returned by the LLM and return the result as a string."""
    name = tool_call.function.name
    args = tool_call.function.arguments  # already a dict with ollama >= 0.2

    if name == "search_web":
        return search_web(
            query=args.get("query", ""),
            max_results=args.get("max_results", 5)
        )

    return f"Unknown tool: {name}"


def run_chat_with_tools(model: str, messages: list) -> tuple[str, dict]:
    """
    Run a chat session with automatic tool-call handling.

    Flow:
      1. Call the LLM with the tool definitions.
      2. If the LLM requests tool calls, execute them and append results.
      3. Call the LLM a second time (no tools offered) for the final answer.
      4. Return (reply_text, raw_response_dict).

    If the LLM answers directly (no tool call), only one LLM call is made.
    """

    # --- First LLM call ---
    response = ollama.chat(
        model=model,
        messages=messages,
        tools=SEARCH_TOOL_DEFINITION,
        options={"think": False}
    )
    msg = response["message"]

    # Qwen3 sometimes bleeds <tool_call> XML into msg.content even when
    # tool_calls is properly populated — treat both cases as a tool request.
    has_tool_calls = bool(msg.tool_calls) or "<tool_call>" in (msg.content or "")

    if not has_tool_calls:
        # LLM answered directly — single call path
        return msg.content, response

    # --- Tool execution ---
    # Serialise the assistant turn as a plain dict for cross-model safety
    extended_messages = messages + [{
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            }
            for tc in (msg.tool_calls or [])
        ]
    }]

    for tool_call in (msg.tool_calls or []):
        result_text = dispatch_tool_call(tool_call)
        print(f"[tool result preview] {result_text[:120]}...")
        extended_messages.append({
            "role": "tool",
            "content": result_text
        })

    print("tool results ready — calling LLM for final answer")

    # --- Second LLM call — no tools offered so it must answer directly ---
    final_response = ollama.chat(
        model=model,
        messages=extended_messages,
        options={"think": False}
    )
    final_msg = final_response["message"]

    return final_msg.content, final_response


# -----------------------------
# FLASK OLLAMA API
# -----------------------------

app = Flask(__name__)
CORS(app)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()

    prompt = data.get("message")
    model = data.get("model")

    if not prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400

    try:
        response = ollama.generate(
            model=model,
            prompt=prompt
        )

        return jsonify({
            "output": response.get("response"),
            "created_at": str(response.get("created_at")),
            "model": response.get("model"),
        })

    except Exception as e:
        print(jsonify({"error": str(e)}), 500)
        return jsonify({"error": str(e)}), 500


@app.route("/talk", methods=["POST"])
def talk():
    data = request.get_json()
    message = data.get("message")
    model = data.get("model")

    if not message:
        return jsonify({"error": "Missing 'message'"}), 400

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": message}]
        )
        msg = response["message"]
        msg_dict = {
            "role": msg.role,
            "content": msg.content,
            "images": msg.images,
            "tool_calls": msg.tool_calls
        }
        return jsonify({
            "reply": msg.content,
            "raw_response": {
                "model": response["model"],
                "created_at": response["created_at"],
                "done": response["done"],
                "done_reason": response["done_reason"],
                "total_duration": response["total_duration"],
                "load_duration": response["load_duration"],
                "prompt_eval_count": response["prompt_eval_count"],
                "prompt_eval_duration": response["prompt_eval_duration"],
                "eval_count": response["eval_count"],
                "eval_duration": response["eval_duration"],
                "message": msg_dict
            }
        })

    except Exception as e:
        print(jsonify({"error": str(e)}), 500)
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    model = data.get("model")
    messages = data.get("messages")
    use_tools = data.get("use_tools")  # set to false to disable web search
    print("got request:", model, "| tools:", use_tools)

    if not messages:
        return jsonify({"error": "Missing 'messages' array"}), 400

    try:
        if use_tools:
            reply, response = run_chat_with_tools(model, messages)
        else:
            response = ollama.chat(
                model=model,
                messages=messages,
                options={"think": False}
            )
            reply = response["message"].content

        msg = response["message"]
        msg_dict = {
            "role": msg.role,
            "content": msg.content,
            "images": msg.images,
            "tool_calls": msg.tool_calls
        }

        print("LLM response:", reply)
        return jsonify({
            "reply": reply,
            "raw_response": {
                "model": response["model"],
                "created_at": response["created_at"],
                "done": response["done"],
                "done_reason": response["done_reason"],
                "total_duration": response["total_duration"],
                "load_duration": response["load_duration"],
                "prompt_eval_count": response["prompt_eval_count"],
                "prompt_eval_duration": response["prompt_eval_duration"],
                "eval_count": response["eval_count"],
                "eval_duration": response["eval_duration"],
                "message": msg_dict
            }
        })

    except Exception as e:
        print(jsonify({"error": str(e)}), 500)
        return jsonify({"error": str(e)}), 500


def get_local_ip():
    """Get LAN IP of PC1."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":
    ensure_firewall_rule()

    pc1_ip = get_local_ip()
    print(f"\nServer starting...")
    print(f"PC1 address: http://{pc1_ip}:{PORT}/chat\n")

    from waitress import serve
    serve(app, host="0.0.0.0", port=int(PORT))