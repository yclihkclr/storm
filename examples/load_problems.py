from mpinets.types import PlanningProblem, ProblemSet
import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "problems",
        type=str,
        default="problems/hybrid_solvable_problems.pkl",
        help="A pickle file of sample problems that follow the PlanningProblem format",
    )
    args = parser.parse_args()
    with open(args.problems, "rb") as f:
        problems = pickle.load(f)

    print(problems)