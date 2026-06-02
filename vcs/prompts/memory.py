"""Subject-registry prompt used by the LensWalk inference workflow."""

SUBJECT_REGISTRY_UPDATE_PROMPT = """
You are responsible for maintaining a `subject_registry` for a video analysis
agent. Analyze the latest tool observation, compare it with the current
registry, and output the consolidated registry.

## Input You Will Receive

- [Current Subject Registry]: the subjects identified so far.
- [History]: previous tool calls and their observation summaries.
- [Current Turn]: the latest tool call, raw observation, and any newly observed
  subjects returned by the visual observer.

## Registry Update Rules

1. Merge entries when the new subject clearly refers to an existing subject.
   Synthesize concise descriptions instead of stacking redundant phrases.
2. Add a new entry when the subject is distinct from all existing subjects.
3. Keep at most 15 subjects after the update. Prioritize subjects observed in
   the current turn, then the most recently observed subjects from history.
4. If there are no subjects in the current registry or latest observation,
   output an empty object.

## Output Specification

Return exactly one JSON object with exactly one top-level key:
`updated_subject_registry`.

```json
{
  "updated_subject_registry": {
    "S1": {
      "description": [
        "concise appearance, role, or state description"
      ],
      "appeared_intervals": [
        "[start_sec, end_sec]"
      ]
    }
  }
}
```
"""
