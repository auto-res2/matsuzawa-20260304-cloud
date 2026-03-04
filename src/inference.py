"""
Inference script for prompt-based experiments with LLM APIs.
"""

import os
import sys
import re
import json
from typing import Dict, List, Any
from omegaconf import DictConfig, OmegaConf
import wandb
from openai import OpenAI

from preprocess import load_gsm8k


def get_prompt_template(method_name: str) -> str:
    """
    Get prompt template for the specified method.

    Args:
        method_name: Name of the method (JAM-CoT, Standard-CoT)

    Returns:
        Prompt template string with {question} placeholder
    """
    if method_name == "JAM-CoT":
        return """You are solving a grade-school math word problem. Follow these strict rules:

1. DIFFICULTY ASSESSMENT: First, output a single line "DIFFICULTY: [E/M/H]" (Easy/Medium/Hard)
   - E (Easy): 3 steps max
   - M (Medium): 6 steps max
   - H (Hard): 9 steps max

2. REASONING STEPS: Each step MUST start with an operation tag:
   [READ] - Extract information from problem
   [SETUP] - Define variables or relationships
   [ARITH] - Perform arithmetic (MUST be in format: a op b = c or expr = value)
   [COMPARE] - Compare values or check conditions
   [CONCLUDE] - Final conclusion

3. ARITHMETIC VERIFICATION: Every [ARITH] step MUST be a verifiable equation.
   Examples:
   - [ARITH] 5 + 3 = 8
   - [ARITH] total_cost = 12 * 4 = 48
   - [ARITH] remaining = 100 - 48 = 52

4. AUDIT: After reasoning, output "AUDIT:" and review each step. If any step is unnecessary or redundant, output "DELETE step X" for each one to remove.

5. FINAL ANSWER: Output only the numeric answer on a line starting with "FINAL_ANSWER:"

Problem: {question}

Now solve:"""

    elif method_name == "Standard-CoT":
        return """Let's think step by step to solve this problem.

Problem: {question}

Solution:"""

    else:
        raise ValueError(f"Unknown method: {method_name}")


def extract_final_answer(response: str, method_name: str) -> float:
    """
    Extract final numeric answer from model response.

    Args:
        response: Model response text
        method_name: Method name for parsing strategy

    Returns:
        Numeric answer as float (NaN if extraction fails)
    """
    # For JAM-CoT, look for FINAL_ANSWER: prefix
    if method_name == "JAM-CoT":
        match = re.search(r"FINAL_ANSWER:\s*([0-9.,]+)", response)
        if match:
            answer_str = match.group(1).replace(",", "")
            try:
                return float(answer_str)
            except ValueError:
                pass

    # For Standard-CoT, look for common answer patterns
    # Try to find the last number in the response
    numbers = re.findall(r"(?:answer is|equals?|=)\s*([0-9.,]+)", response.lower())
    if numbers:
        answer_str = numbers[-1].replace(",", "")
        try:
            return float(answer_str)
        except ValueError:
            pass

    # Fallback: extract last number
    all_numbers = re.findall(r"([0-9.,]+)", response)
    if all_numbers:
        answer_str = all_numbers[-1].replace(",", "")
        try:
            return float(answer_str)
        except ValueError:
            pass

    return float("nan")


# [VALIDATOR FIX - Attempt 1]
# [PROBLEM]: All arithmetic steps marked as invalid (faithfulness_rate=0.0)
# [CAUSE]: Original verifier only accepted equations with exactly 2 parts when split by '=',
#          but model outputs multi-part equations like "a = expr = value" (3+ parts).
#          Also, verifier only checked for presence of numbers, not correctness.
# [FIX]: Accept equations with 2+ parts, extract rightmost value as expected result,
#        evaluate arithmetic expression(s) safely, and compare computed vs expected.
#
# [OLD CODE]:
# def verify_arithmetic_steps(response: str) -> Dict[str, Any]:
#     arith_lines = re.findall(r"\[ARITH\](.+?)(?=\n|$)", response)
#     total_steps = len(arith_lines)
#     valid_steps = 0
#     invalid_steps = 0
#     for line in arith_lines:
#         line = line.strip()
#         if "=" in line:
#             try:
#                 parts = line.split("=")
#                 if len(parts) == 2:  # <-- This rejects multi-part equations
#                     lhs = parts[0].strip()
#                     rhs = parts[1].strip()
#                     lhs_nums = re.findall(r"[0-9.]+", lhs)
#                     rhs_nums = re.findall(r"[0-9.]+", rhs)
#                     if lhs_nums and rhs_nums:  # <-- Only checks presence, not correctness
#                         valid_steps += 1
#                     else:
#                         invalid_steps += 1
#                 else:
#                     invalid_steps += 1
#             except Exception:
#                 invalid_steps += 1
#         else:
#             invalid_steps += 1
#     return {...}
#
# [NEW CODE]:
def verify_arithmetic_steps(response: str) -> Dict[str, Any]:
    """
    Verify arithmetic steps in JAM-CoT response.

    Args:
        response: Model response text

    Returns:
        Dictionary with verification metrics
    """
    arith_lines = re.findall(r"\[ARITH\](.+?)(?=\n|$)", response)

    total_steps = len(arith_lines)
    valid_steps = 0
    invalid_steps = 0

    for line in arith_lines:
        line = line.strip()
        # Try to parse equation
        if "=" in line:
            try:
                # Extract equation parts
                parts = line.split("=")
                if len(parts) >= 2:  # Accept 2 or more parts
                    # The rightmost part is the claimed answer
                    expected_str = parts[-1].strip()
                    # Remove any trailing punctuation
                    expected_str = re.sub(r"[^\d.\-+*/() ]", "", expected_str)

                    # Try to find a pure arithmetic expression to verify
                    # Look for the part that contains numeric operations
                    arithmetic_expr = None
                    for part in parts[:-1]:
                        # Check if this part contains arithmetic operations with numbers
                        if re.search(r"\d+\s*[\+\-\*/]\s*\d+", part):
                            arithmetic_expr = part.strip()
                            break

                    # If no arithmetic expression found, check if we have a simple assignment
                    # like "white_fiber = 1" where the value is already computed
                    if arithmetic_expr is None:
                        # Check if second-to-last part is just a number
                        if len(parts) >= 2:
                            prev_part = parts[-2].strip()
                            # Remove variable names and equals signs
                            prev_part = re.sub(r"[a-zA-Z_]+\s*", "", prev_part).strip()
                            if prev_part and re.match(r"^[\d.\-+*/() ]+$", prev_part):
                                arithmetic_expr = prev_part

                    # Try to evaluate and compare
                    if arithmetic_expr:
                        # Clean the expression: remove variable names, keep only numbers and operators
                        clean_expr = re.sub(r"[a-zA-Z_]+", "", arithmetic_expr)
                        clean_expr = clean_expr.strip()

                        # Safe evaluation: only allow basic arithmetic
                        if clean_expr and re.match(r"^[\d.\-+*/() ]+$", clean_expr):
                            try:
                                computed = eval(clean_expr, {"__builtins__": {}}, {})
                                expected = eval(expected_str, {"__builtins__": {}}, {})

                                # Compare with small tolerance for floating point
                                if abs(computed - expected) < 0.01:
                                    valid_steps += 1
                                else:
                                    invalid_steps += 1
                            except Exception:
                                invalid_steps += 1
                        else:
                            invalid_steps += 1
                    else:
                        # No arithmetic found, but format is okay - count as valid
                        # This handles cases like "variable = 5" where value is given
                        try:
                            float(expected_str)
                            valid_steps += 1
                        except ValueError:
                            invalid_steps += 1
                else:
                    invalid_steps += 1
            except Exception:
                invalid_steps += 1
        else:
            invalid_steps += 1

    return {
        "total_arith_steps": total_steps,
        "valid_arith_steps": valid_steps,
        "invalid_arith_steps": invalid_steps,
        "faithfulness_rate": valid_steps / total_steps if total_steps > 0 else 1.0,
    }


def run_inference(cfg: DictConfig):
    """
    Run inference for a single configuration.

    Args:
        cfg: Hydra configuration
    """
    # Determine number of samples based on mode
    if cfg.mode == "sanity_check":
        num_samples = cfg.run.dataset.num_samples_sanity
        wandb_project = f"{cfg.wandb.project}-sanity"
    else:
        num_samples = cfg.run.dataset.num_samples
        wandb_project = cfg.wandb.project

    # Initialize WandB
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=wandb_project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Load dataset
    print(f"Loading {num_samples} examples from GSM8K...")
    examples = load_gsm8k(
        split=cfg.run.dataset.split, num_samples=num_samples, cache_dir=".cache"
    )

    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Get prompt template
    method_name = cfg.run.method.name
    prompt_template = get_prompt_template(method_name)

    # Run inference
    results = []
    correct = 0
    total = 0

    # Metrics for JAM-CoT
    total_faithfulness = 0.0
    num_with_arith = 0

    print(f"Running inference with {method_name}...")
    for i, example in enumerate(examples):
        question = example["question"]
        gold_answer = example["answer"]

        # Create prompt
        prompt = prompt_template.format(question=question)

        # Call API
        response = client.chat.completions.create(
            model=cfg.run.model.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.run.model.temperature,
            max_tokens=cfg.run.model.max_tokens,
        )

        response_text = response.choices[0].message.content

        # Extract answer
        predicted_answer = extract_final_answer(response_text, method_name)

        # Check correctness (allow small floating point error)
        is_correct = abs(predicted_answer - gold_answer) < 0.01
        if is_correct:
            correct += 1
        total += 1

        # Verify arithmetic for JAM-CoT
        arith_metrics = {}
        if method_name == "JAM-CoT":
            arith_metrics = verify_arithmetic_steps(response_text)
            if arith_metrics["total_arith_steps"] > 0:
                total_faithfulness += arith_metrics["faithfulness_rate"]
                num_with_arith += 1

        # Store result
        result = {
            "example_id": i,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "response": response_text,
            **arith_metrics,
        }
        results.append(result)

        # Log to WandB
        if cfg.wandb.mode != "disabled":
            wandb.log(
                {
                    "example_id": i,
                    "accuracy": correct / total,
                    "is_correct": int(is_correct),
                    **(
                        {f"arith_{k}": v for k, v in arith_metrics.items()}
                        if arith_metrics
                        else {}
                    ),
                }
            )

        if (i + 1) % 10 == 0:
            print(
                f"Processed {i + 1}/{len(examples)} examples, accuracy: {correct / total:.3f}"
            )

    # Compute final metrics
    accuracy = correct / total
    metrics = {"accuracy": accuracy, "num_correct": correct, "num_total": total}

    if method_name == "JAM-CoT" and num_with_arith > 0:
        avg_faithfulness = total_faithfulness / num_with_arith
        metrics["avg_faithfulness_rate"] = avg_faithfulness

    print(f"\nFinal Accuracy: {accuracy:.4f}")
    if "avg_faithfulness_rate" in metrics:
        print(f"Avg Faithfulness Rate: {metrics['avg_faithfulness_rate']:.4f}")

    # Save results
    results_dir = os.path.join(cfg.results_dir, cfg.run.run_id)
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Log final metrics to WandB
    if cfg.wandb.mode != "disabled":
        for k, v in metrics.items():
            wandb.summary[k] = v
        print(f"\nWandB run: {wandb.run.url}")
        wandb.finish()

    # Sanity validation
    if cfg.mode == "sanity_check":
        run_sanity_validation(results, metrics, method_name)


def run_sanity_validation(results: List[Dict], metrics: Dict, method_name: str):
    """
    Run sanity validation checks and print verdict.

    Args:
        results: List of inference results
        metrics: Aggregated metrics
        method_name: Method name
    """
    # Check: at least 5 samples processed
    num_samples = len(results)
    if num_samples < 5:
        print(
            f"SANITY_VALIDATION: FAIL reason=insufficient_samples (got {num_samples}, need 5)"
        )
        return

    # Check: all outputs are valid (predicted_answer is not NaN)
    valid_outputs = sum(
        1 for r in results if not (r["predicted_answer"] != r["predicted_answer"])
    )  # NaN check
    if valid_outputs < num_samples:
        print(
            f"SANITY_VALIDATION: FAIL reason=invalid_outputs (valid={valid_outputs}, total={num_samples})"
        )
        return

    # Check: not all outputs are identical
    unique_answers = len(set(r["predicted_answer"] for r in results))
    if unique_answers == 1 and num_samples > 1:
        print(f"SANITY_VALIDATION: FAIL reason=all_identical_outputs")
        return

    # Check: accuracy is not 0 (at least one correct)
    if metrics["num_correct"] == 0:
        print(f"SANITY_VALIDATION: FAIL reason=zero_accuracy")
        return

    # All checks passed
    print("SANITY_VALIDATION: PASS")

    # Print summary
    summary = {
        "samples": num_samples,
        "accuracy": metrics["accuracy"],
        "correct": metrics["num_correct"],
        "valid_outputs": valid_outputs,
        "unique_answers": unique_answers,
    }

    if method_name == "JAM-CoT" and "avg_faithfulness_rate" in metrics:
        summary["faithfulness_rate"] = metrics["avg_faithfulness_rate"]

    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")


if __name__ == "__main__":
    print("This script should be called from src/main.py", file=sys.stderr)
    sys.exit(1)
