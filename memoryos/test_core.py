from memoryos.core import MemoryOS

memory = MemoryOS(
    db_path="memoryos_test.db",
    session_id="test_session",
)

memory.clear_all()

result_1 = memory.process_turn(
    user_message=(
        "My name is Aryan. "
        "I prefer dark UI. "
        "I want to build MemoryOS."
    ),
    ai_response="Nice, MemoryOS sounds powerful.",
)

result_2 = memory.process_turn(
    user_message=(
        "I am working on an AI memory system. "
        "I prefer dark UI."
    ),
    ai_response="Got it.",
)

print("\nNew facts from turn 1:")
for fact in result_1["new_facts"]:
    print("-", fact.content)

print("\nNew facts from turn 2:")
for fact in result_2["new_facts"]:
    print("-", fact.content)

print("\nAll stored facts:")
for fact in memory.get_all_facts():
    print("-", fact.content, "| embedding:", fact.embedding is not None)

print("\nSearch: what UI does the user like?")
results = memory.search_memory("What UI theme does the user prefer?", top_k=3)

for result in results:
    print(round(result.score, 4), "-", result.content)

print("\nSearch: what is the user building?")
results = memory.search_memory("What project is the user building?", top_k=3)

for result in results:
    print(round(result.score, 4), "-", result.content)

print("\nGenerated LLM context:")
context = memory.build_context(
    "What should I know about the user's project and preferences?",
    limit=5,
)

print(context)