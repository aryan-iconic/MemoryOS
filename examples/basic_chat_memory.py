from memoryos import MemoryOS

memory = MemoryOS(db_path="chat_memory.db", session_id="demo")

# Simulate a conversation
memory.process_turn("My name is Aryan. I prefer dark UI.", "Got it!")
memory.process_turn("I'm building an AI memory library.", "That's cool!")

# Later — retrieve context
context = memory.build_context("What do you know about me?")
print(context)

# Search memory directly
results = memory.search_memory("dark UI")
for r in results:
    print(r.content, r.score)

memory.close()