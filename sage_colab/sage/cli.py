from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="sage", description="SAGE local diagnostics and hosted-Colab agent")
    parser.add_argument("--state", default="./sage_state")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train-world", help="Bounded resumable world/policy/controller bootstrap")
    train.add_argument("--steps", type=int, default=120)
    collect = sub.add_parser("collect", help="Collect observed simulator transitions")
    collect.add_argument("--episodes", type=int, default=20)
    collect.add_argument("--output", default="trajectories.jsonl")
    sub.add_parser("evaluate", help="Evaluate calibration, retention and simulated transfer")
    sub.add_parser("status", help="List persisted goals and training reports")
    run = sub.add_parser("run", help="Run an owner goal; LLM is restricted to hosted Colab")
    run.add_argument("goal")
    run.add_argument("--output", default="")
    run.add_argument("--model", default="auto")
    args = parser.parse_args()
    state = Path(args.state)
    if args.command == "collect":
        from .simulators import collect_dataset
        rows = collect_dataset("train", episodes_per_family=max(1, min(args.episodes, 100)), seed=17)
        Path(args.output).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(json.dumps({"rows": len(rows), "output": args.output, "provenance": "validated_simulation"}))
    elif args.command in {"train-world", "evaluate"}:
        from .training import LearningSystem
        model = LearningSystem(state)
        result = model.bootstrap(args.steps) if args.command == "train-world" else model.evaluate()
        print(json.dumps(result, indent=2))
    elif args.command == "status":
        for path in sorted(state.rglob("*report.json")) + sorted((state / "runs").glob("*.json")):
            print(path)
            print(path.read_text()[:6000])
    elif args.command == "run":
        from .authority import AuthorityService
        from .memory import MemoryStore
        from .training import LearningSystem
        from .llm import ColabLLM
        from .agent import SageAgent
        memory = MemoryStore(state / "memory")
        with AuthorityService(state / "authority", state / "workspace") as authority:
            learning = LearningSystem(state)
            llm = ColabLLM(state, args.model)
            llm.load()
            agent = SageAgent(state, authority, memory, learning, llm)
            completion = {"type": "artifact", "path": args.output} if args.output else {"type": "answer"}
            goal = agent.create(args.goal, completion)
            result = agent.run(goal, emit=print)
            print(json.dumps({"status": result["status"], "answer": result["answer"]}, indent=2))
        memory.close()


if __name__ == "__main__":
    main()
