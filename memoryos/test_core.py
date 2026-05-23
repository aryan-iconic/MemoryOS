from memoryos.core import MemoryOS

memory = MemoryOS(
    db_path="memoryos_test.db",
    session_id="test_session",
)

memory.clear_all()

memory.process_turn(
    user_message="My name is Aryan.",
    ai_response="Nice to meet you, Aryan.",
)

memory.process_turn(
    user_message="I prefer dark UI.",
    ai_response="Got it. I will remember your UI preference.",
)

memory.process_turn(
    user_message="I want to build MemoryOS.",
    ai_response="That sounds like a powerful project.",
)

print("\nWorking memory context:")
print(memory.working_memory.build_context())

print("\nFinal prompt context:")
print(
    memory.build_prompt_context(
        query="What should I know about the user?",
    )
)