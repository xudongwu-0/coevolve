"""Local WHALE-style Lichess replication; the reference verifier is never editable."""
from __future__ import annotations

import asyncio
import json
import re
import time

import chess

from .program_harness import EDITOR_API_GUIDE, HookExecutionError, ProgramValidationError
from .retool import ContextExhausted

ATTRIBUTES = {"MAX_TURNS", "SYSTEM_PROMPT", "FORMAT_RETRY_BUDGET", "ILLEGAL_MOVE_RETRY_BUDGET"}
HOOKS = {"format_observation", "parse_action", "recover_action"}
MAX_RESPONSE_CHARS = 32768
MAX_RESPONSE_JSON_BYTES = 65536
MAX_OBSERVATION_CHARS = 16384
MAX_RECOVERY_CHARS = 4096
MAX_HARNESS_ERROR_CHARS = 2048
UCI = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?")

INITIAL_SOURCE = '''class CandidateHarness:
    MAX_TURNS = 9
    SYSTEM_PROMPT = "Solve the chess puzzle one legal move at a time using the visible board. Return exactly <move>uci</move>, for example <move>e2e4</move>. No explanations or tools."
    FORMAT_RETRY_BUDGET = 1
    ILLEGAL_MOVE_RETRY_BUDGET = 1
'''

EDITOR_SYSTEM = """Improve a reusable chess-puzzle harness, not a puzzle answer.
Return ONLY a Python class CandidateHarness, with no bases, decorators,
top-level imports/code, or additional attributes. Start from:
""" + INITIAL_SOURCE + """
All four literal settings are required: MAX_TURNS is an integer 1-18;
SYSTEM_PROMPT is a nonempty string up to 1600 characters; FORMAT_RETRY_BUDGET
and ILLEGAL_MOVE_RETRY_BUDGET are integers 0-3, each a whole-episode retry budget.
Optionally define any of these methods, each with signature (self, observation):
  format_observation: return a nonempty string (at most 16384 characters).
  parse_action: return one lowercase UCI string (4-5 characters), or None.
  recover_action: return a nonempty retry message (at most 4096 characters),
                  or None to stop after a malformed/illegal action.
These are executable hooks, not just prompts: change board presentation, map a
unique SAN/JSON/other response to UCI using visible legal_moves, or implement
bounded recovery. Omitted hooks use trusted defaults; there is no automatic move
selection. Hooks run in isolation, without files/network or persistent state;
only the hook-local imports and APIs described below are available.
Invalid outputs, exceptions, or timeouts in implemented hooks immediately
end the trajectory with zero reward and candidate_execution_error; no fallback.
Trusted defaults apply ONLY when a hook is omitted. Source limit: 32000 UTF-8
bytes; hook input limit: 128000 serialized JSON bytes; isolated hook result limit:
65536 JSON bytes; wall deadline: 15 seconds. Model response limit: 32768 characters
and 65536 JSON bytes. These fixed limits cannot be edited.
If formatting fails before generation, a fixed terminal error-only policy turn
(at most 128 tokens, no board or solving prompt) records that failed trajectory.
Its reward is always zero regardless of the model's diagnostic response.

PUBLIC observation is a JSON object with exactly:
fen: string; board_ascii: string (rank 8 to 1, uppercase White, lowercase Black);
side_to_move: "white"|"black"; legal_moves: list of {"uci": str, "san": str};
last_opponent_move: string|null; accepted_solver_moves: list[str];
solver_move_number: int (1-based); turn: int (1-based generation call);
turns_remaining: int (including this call); remaining_tokens: int;
response: string|null (model response for parsing/recovery, otherwise null);
error: null|"invalid_format"|"illegal_move";
parsed_action: string|null; retries_remaining: int (matching error budget).
Do not rely on absent private puzzle IDs, ratings, themes, reference moves, or
solution length. Defaults show FEN/ASCII/legal UCI+SAN moves, parse exactly one
<move>uci</move> (ignoring an optional completed <think> block), and retry bad
format/illegal moves while budget remains. Custom parsing still must produce
exactly one syntactically valid UCI. The fixed native python-chess verifier
checks legality AND exact reference continuation, plays the reference opponent
reply, and awards 1 only for the complete line. A legal but wrong move ends the
episode immediately and never reveals the correct move. You cannot edit reward,
reference, opponent, token budget, action size bounds, or verifier.
This is a local replication with Qwen3-4B, not an exact-paper score reproduction.
""" + EDITOR_API_GUIDE


class CandidateExecutionError(ValueError):
    """Generated hook failure: a failed trajectory, never an infrastructure halt."""


def harness_spec(source: str) -> dict:
    from .program_harness import validate_program

    spec = validate_program(source, attributes=ATTRIBUTES, hooks=HOOKS)
    if type(spec["MAX_TURNS"]) is not int or not 1 <= spec["MAX_TURNS"] <= 18:
        raise ValueError("MAX_TURNS must be an integer in 1..18")
    for name in ("FORMAT_RETRY_BUDGET", "ILLEGAL_MOVE_RETRY_BUDGET"):
        if type(spec[name]) is not int or not 0 <= spec[name] <= 3:
            raise ValueError(f"{name} must be an integer in 0..3")
    if not isinstance(spec["SYSTEM_PROMPT"], str) or not 1 <= len(spec["SYSTEM_PROMPT"]) <= 1600:
        raise ValueError("SYSTEM_PROMPT must contain 1..1600 characters")
    return spec


def position_key(fen: str) -> str:
    """Deliberately ignore clocks, castling and en-passant for conservative dedup."""
    board = chess.Board(fen)
    return board.board_fen() + (" w" if board.turn else " b")


def validate_reference(row: dict) -> tuple[chess.Board, list[str]]:
    board = chess.Board(row["start_fen"])
    if not board.is_valid():
        raise ValueError("Invalid puzzle start position")
    line = row["solution_moves"]
    if not isinstance(line, list) or not 1 <= len(line) <= 9:
        raise ValueError("Reference must contain 1..9 plies")
    verify = board.copy()
    for uci in line:
        if not isinstance(uci, str) or not UCI.fullmatch(uci):
            raise ValueError("Invalid reference UCI")
        move = chess.Move.from_uci(uci)
        if move not in verify.legal_moves:
            raise ValueError("Illegal reference continuation")
        verify.push(move)
    return board, list(line)


def board_ascii(board: chess.Board) -> str:
    rows = [
        f"{rank + 1} " + " ".join(
            piece.symbol() if (piece := board.piece_at(chess.square(file, rank))) else "."
            for file in range(8)
        )
        for rank in range(7, -1, -1)
    ]
    return "\n".join(rows + ["  a b c d e f g h"])


def format_observation(observation: dict) -> str:
    lines = [
        "Solve the chess puzzle one move at a time.",
        f"Side to move: {observation['side_to_move']}",
        f"FEN: {observation['fen']}", "Board:", observation["board_ascii"],
        f"Puzzle move number: {observation['solver_move_number']}",
    ]
    if observation["last_opponent_move"]:
        lines.append("Last opponent reply: " + observation["last_opponent_move"])
    if observation["accepted_solver_moves"]:
        lines.append("Your accepted moves: " + ", ".join(observation["accepted_solver_moves"]))
    lines.append("Legal moves:")
    lines.extend(f"- [{move['uci']}] SAN={move['san']}" for move in observation["legal_moves"])
    lines.append("Return exactly one move as <move>uci</move>.")
    return "\n".join(lines)


def parse_action(observation: dict) -> str | None:
    response = observation["response"] or ""
    # Thinking may mention other moves, but the final answer must be unambiguous.
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[1]
    match = re.fullmatch(r"\s*<move>\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*</move>\s*", response, re.I)
    return match.group(1).lower() if match else None


def recover_action(observation: dict) -> str:
    if observation["error"] == "illegal_move":
        return "That move is illegal. Choose one visible legal move and reply only as <move>uci</move>."
    return "The response must specify exactly one move. Reply only as <move>uci</move>."


def _execute_hook(spec: dict, hook: str, observation: dict):
    from .program_harness import execute_hook
    return execute_hook(spec, hook, observation)


async def _hook(spec: dict, name: str, observation: dict, errors: list[dict]):
    default = {"format_observation": format_observation,
               "parse_action": parse_action, "recover_action": recover_action}[name]
    if name not in spec.get("_hooks", []):
        return default(observation)
    try:
        result = await asyncio.to_thread(_execute_hook, spec, name, observation)
        if name == "parse_action":
            if result is not None and (not isinstance(result, str) or not UCI.fullmatch(result)):
                raise CandidateExecutionError("parse_action must return one UCI or null")
        elif result is None and name == "recover_action":
            return None
        else:
            limit = MAX_OBSERVATION_CHARS if name == "format_observation" else MAX_RECOVERY_CHARS
            if not isinstance(result, str) or not result.strip() or len(result) > limit:
                raise CandidateExecutionError("Invalid hook text output")
        return result
    except (CandidateExecutionError, HookExecutionError, ProgramValidationError, TimeoutError) as exc:
        reason = f"{type(exc).__name__}: {exc}"[:MAX_HARNESS_ERROR_CHARS]
        errors.append({"hook": name, "error": type(exc).__name__, "reason": reason,
                       "turn": observation["turn"]})
        raise CandidateExecutionError(f"Generated hook failed: {name}") from exc


async def run_episode(row: dict, spec: dict, generate, total_tokens=4096, seed=42) -> dict:
    """Keep every generation record; only public board state crosses hook boundaries."""
    if type(total_tokens) is not int or total_tokens < 1:
        raise ValueError("total_tokens must be a positive integer")
    board, reference = validate_reference(row)
    started = time.monotonic()
    messages = [{"role": "system", "content": spec["SYSTEM_PROMPT"]}]
    calls, events, hook_errors, accepted, opponents = [], [], [], [], []
    remaining, index, generation_calls = total_tokens, 0, 0
    format_errors = illegal_moves = wrong_moves = tool_calls = tool_errors = 0
    format_retries = illegal_retries = 0
    last_opponent = None
    reward, valid_format, completion_truncated = 0.0, False, False
    stop_reason = "turn_limit"
    failure_reason = None
    diagnostic_error_turn = False
    retry_message = None

    for turn in range(spec["MAX_TURNS"]):
        observation = {
            "fen": board.fen(), "board_ascii": board_ascii(board),
            "side_to_move": "white" if board.turn else "black",
            "legal_moves": sorted(
                [{"uci": move.uci(), "san": board.san(move)} for move in board.legal_moves],
                key=lambda move: move["uci"],
            ),
            "last_opponent_move": last_opponent,
            "accepted_solver_moves": list(accepted), "solver_move_number": len(accepted) + 1,
            "turn": turn + 1, "turns_remaining": spec["MAX_TURNS"] - turn,
            "remaining_tokens": remaining, "response": None, "error": None,
            "parsed_action": None, "retries_remaining": 0,
        }
        try:
            rendered = await _hook(spec, "format_observation", observation, hook_errors)
        except CandidateExecutionError:
            stop_reason, valid_format = "candidate_execution_error", False
            events.append({"turn": turn + 1, "error": stop_reason, "hook": "format_observation"})
            failure_reason = hook_errors[-1]["reason"]
            if generation_calls == 0:
                # No candidate prompt, board, exception text, or trusted solve fallback.
                messages[:] = [
                    {"role": "system", "content": (
                        "A harness program failed. This is a terminal diagnostic turn, not a "
                        "puzzle-solving task. Reply only: Harness execution failed."
                    )},
                    {"role": "user", "content": (
                        "format_observation failed before an observation was shown. "
                        "Acknowledge the failure; do not propose an action."
                    )},
                ]
                cap = min(128, remaining)
                completion = await generate(messages, [], cap, seed + turn)
                generation_calls += 1
                diagnostic_error_turn = True
                if "record" in completion:
                    calls.append(completion["record"])
                used = completion["usage"]["completion_tokens"]
                if type(used) is not int or not 0 < used <= cap:
                    raise RuntimeError("Invalid diagnostic generation token accounting")
                remaining -= used
                choice = completion["choices"][0]
                completion_truncated |= choice.get("finish_reason") == "length"
                message = dict(choice["message"])
                if message.get("role", "assistant") != "assistant":
                    raise RuntimeError("Expected assistant diagnostic generation")
                message["role"] = "assistant"
                messages.append(message)
            break
        messages.append({"role": "user", "content": (
            retry_message + "\n\n" + rendered if retry_message else rendered
        )})
        retry_message = None
        cap = min(remaining, max(256, total_tokens // 5))
        try:
            completion = await generate(messages, [], cap, seed + turn)
        except ContextExhausted:
            stop_reason = "context_limit"
            break
        generation_calls += 1
        if "record" in completion:
            calls.append(completion["record"])
        used = completion["usage"]["completion_tokens"]
        if type(used) is not int or not 0 < used <= cap:
            raise RuntimeError("Invalid generation token accounting")
        remaining -= used
        choice = completion["choices"][0]
        completion_truncated |= choice.get("finish_reason") == "length"
        message = dict(choice["message"])
        if message.get("role", "assistant") != "assistant":
            raise RuntimeError("Expected assistant generation")
        message["role"] = "assistant"
        messages.append(message)
        response = message.get("content") or ""
        valid_format = False
        if message.get("tool_calls"):
            tool_calls += len(message["tool_calls"])
            tool_errors += len(message["tool_calls"])
            format_errors += 1
            stop_reason = "unsupported_tool"
            events.append({"turn": turn + 1, "error": stop_reason})
            break
        if (not isinstance(response, str) or len(response) > MAX_RESPONSE_CHARS
                or len(json.dumps(response, ensure_ascii=True).encode()) > MAX_RESPONSE_JSON_BYTES):
            format_errors += 1
            stop_reason = "response_too_long" if isinstance(response, str) else "invalid_format"
            events.append({"turn": turn + 1, "error": stop_reason})
            break
        observation.update(response=response, remaining_tokens=remaining)
        try:
            action = await _hook(spec, "parse_action", observation, hook_errors)
        except CandidateExecutionError:
            stop_reason = "candidate_execution_error"
            events.append({"turn": turn + 1, "error": stop_reason, "hook": "parse_action"})
            failure_reason = hook_errors[-1]["reason"]
            break
        valid_format = isinstance(action, str) and UCI.fullmatch(action) is not None
        observation["parsed_action"] = action
        if not valid_format:
            format_errors += 1
            stop_reason = "invalid_format"
            retry_left = spec["FORMAT_RETRY_BUDGET"] - format_retries
        else:
            move = chess.Move.from_uci(action) if action[:2] != action[2:4] else None
            if move is None or move not in board.legal_moves:
                illegal_moves += 1
                stop_reason = "illegal_move"
                retry_left = spec["ILLEGAL_MOVE_RETRY_BUDGET"] - illegal_retries
            elif action != reference[index]:
                wrong_moves += 1
                stop_reason = "wrong_move"
                events.append({"turn": turn + 1, "error": stop_reason, "action": action})
                break
            else:
                board.push(move)
                accepted.append(action)
                index += 1
                if index < len(reference):
                    reply = chess.Move.from_uci(reference[index])
                    if reply not in board.legal_moves:
                        raise ValueError("Reference opponent move became illegal")
                    board.push(reply)
                    last_opponent = reply.uci()
                    opponents.append(last_opponent)
                    index += 1
                if index == len(reference):
                    reward, stop_reason = 1.0, "solved"
                    break
                if not remaining:
                    stop_reason = "token_limit"
                    break
                stop_reason = "turn_limit"
                continue
        events.append({"turn": turn + 1, "error": stop_reason, "action": action})
        if retry_left <= 0:
            break
        if not remaining:
            stop_reason = "token_limit"
            break
        if turn + 1 >= spec["MAX_TURNS"]:
            stop_reason = "turn_limit"
            break
        observation.update(error=stop_reason, retries_remaining=retry_left)
        try:
            retry_message = await _hook(spec, "recover_action", observation, hook_errors)
        except CandidateExecutionError:
            stop_reason, valid_format = "candidate_execution_error", False
            events.append({"turn": turn + 1, "error": stop_reason, "hook": "recover_action"})
            failure_reason = hook_errors[-1]["reason"]
            break
        if retry_message is None:
            break
        if stop_reason == "invalid_format":
            format_retries += 1
        else:
            illegal_retries += 1
        stop_reason = "turn_limit"

    return {
        "id": row["id"], "reward": reward, "messages": messages, "calls": calls,
        "tokens": total_tokens - remaining,
        "truncated": stop_reason in {"token_limit", "turn_limit", "context_limit"},
        "boxed": valid_format, "tool_calls": tool_calls, "tool_errors": tool_errors,
        "elapsed_seconds": time.monotonic() - started,
        "domain": "chess", "outcome": "solved" if reward else "failed",
        "stop_reason": stop_reason, "generation_calls": generation_calls,
        "valid_action_format": valid_format, "completion_truncated": completion_truncated,
        "format_errors": format_errors, "illegal_moves": illegal_moves, "wrong_moves": wrong_moves,
        "failed_actions": format_errors + illegal_moves + wrong_moves,
        "format_retries": format_retries, "illegal_retries": illegal_retries,
        "accepted_solver_moves": accepted, "opponent_moves": opponents, "plies_played": index,
        "final_fen": board.fen(), "failure_trace": events, "hook_errors": hook_errors,
        "harness_errors": hook_errors, "harness_error": bool(hook_errors),
        "failure_reason": failure_reason or (stop_reason if not reward else None),
        "diagnostic_error_turn": diagnostic_error_turn,
    }
