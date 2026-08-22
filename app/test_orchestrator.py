from .orchestrator import run_orchestrator

user_input = input("Teacher: ")

result = run_orchestrator(user_input)

print(result)