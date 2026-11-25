from deid import deidentify
from llm_check import check_ollama, llm_find_pii

check_ollama()

tests = [
    "I'm the only transfer student from Izmir in section B, so the TA will know it's me.",
    "jordy said nakamuraSolution() runs faster than mine",
    "Hi, I'm Jordan Nakamura and my roommate Ahmet Yilmaz helped me",
    "def dijkstra(graph, start):  # standard shortest path",
    "for user in users:\n    print(user.name, user.email)",
]
for t in tests:
    masked = deidentify(t)
    print("RULES:", masked)
    findings = llm_find_pii(masked)
    for f in findings:
        print(f"  LLM -> {f['text']!r} [{f['type']}] {f['reason']}")
    if not findings:
        print("  LLM -> nothing")
    print()