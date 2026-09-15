"""CLI entry point for the AI Learning & Study Assistant.

Run with: python -m src.main
"""
import ollama

from src.agent import Agent
from src.rag.ingest import ingest_file
from src.tools.quiz import generate_quiz
from src.memory import StudentMemory
from src.config import OLLAMA_HOST, OLLAMA_MODEL


def _check_ollama() -> bool:
    """Return True if Ollama is reachable and the configured model is pulled."""
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        models = [m["model"] for m in client.list().get("models", [])]
        if not any(OLLAMA_MODEL in m for m in models):
            print(
                f"Warning: model '{OLLAMA_MODEL}' not found in Ollama. "
                f"Run: ollama pull {OLLAMA_MODEL}\n"
            )
            return False
        return True
    except Exception as e:
        print(
            f"Warning: could not reach Ollama at {OLLAMA_HOST} ({e}).\n"
            "Make sure Ollama is installed and running (`ollama serve`), "
            f"and that you've pulled a model (`ollama pull {OLLAMA_MODEL}`).\n"
        )
        return False


def print_quiz(quiz: dict) -> None:
    if quiz.get("error"):
        print(f"! {quiz['error']}")
        return
    print(f"\nQuiz: {quiz['topic']}")
    for i, q in enumerate(quiz.get("questions", []), start=1):
        print(f"\nQ{i}. {q['question']}")
        if q["type"] == "multiple_choice":
            for opt in q["options"]:
                print(f"   - {opt}")
        print(f"   (answer: {q['answer']})")
        if q.get("explanation"):
            print(f"   why: {q['explanation']}")


def main():
    _check_ollama()

    student_id = input("Student ID (e.g. demo_student): ").strip() or "demo_student"
    agent = Agent(student_id)

    print(f"\nWelcome, {student_id}.")
    print(agent.memory.summary())
    print(
        "\nCommands: /upload <path>  /plan <exam_date YYYY-MM-DD> <hours_per_week>  "
        "/quiz <topic>  /progress  /quit\n"
    )

    while True:
        try:
            message = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not message:
            continue
        if message == "/quit":
            print("Goodbye!")
            break

        if message.startswith("/upload "):
            path = message.split(" ", 1)[1].strip()
            try:
                n = ingest_file(student_id, path)
                print(f"Ingested {n} chunks from {path}")
            except Exception as e:
                print(f"! Could not ingest file: {e}")
            continue

        if message.startswith("/quiz "):
            topic = message.split(" ", 1)[1].strip()
            quiz = generate_quiz(student_id, topic)
            print_quiz(quiz)
            continue

        if message == "/progress":
            memory = StudentMemory.load(student_id)
            print(memory.summary())
            continue

        if message.startswith("/plan "):
            # Delegate to the agent so it can pick sensible topics from the
            # ingested materials via the create_learning_plan tool.
            reply = agent.handle(
                f"Create a learning plan. Details: {message[len('/plan '):]}"
            )
            print(reply)
            continue

        # Free-form question -> agent decides whether to retrieve, quiz, etc.
        reply = agent.handle(message)
        print(reply)


if __name__ == "__main__":
    main()
