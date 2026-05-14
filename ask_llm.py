import json
from pathlib import Path
import litellm

# Khởi tạo model Qwen2.5 local
llm = litellm.LLM(model="ollama/qwen2.5:3b")  # LiteLLM mới API

# Load tree structure
tree_file = Path("./results/data_structure.json")
with tree_file.open("r", encoding="utf-8") as f:
    tree = json.load(f)

# Traverse tree để lấy node liên quan
def retrieve_nodes(node, question, path=""):
    nodes_found = []
    if isinstance(node, dict):
        title = node.get("title", "")
        summary = node.get("summary", "")
        text = node.get("text", "")
        combined = f"{title} {summary} {text}".lower()
        new_path = f"{path} > {title}" if title else path
        if question.lower() in combined:
            nodes_found.append({"path": new_path, "text": combined})
        for child in node.get("nodes", []):
            nodes_found.extend(retrieve_nodes(child, question, new_path))
    elif isinstance(node, list):
        for item in node:
            nodes_found.extend(retrieve_nodes(item, question, path))
    return nodes_found

question = input("Question: ").strip()
retrieved = retrieve_nodes(tree, question)
if not retrieved:
    print("No relevant nodes found.")
    exit()

context = "\n\n".join([f"{n['path']}\n{n['text']}" for n in retrieved])
prompt = f"Question: {question}\nContext:\n{context}"

# Gọi LLM reasoning
answer = llm.generate(prompt)

print("\n=== Answer ===\n")
print(answer)