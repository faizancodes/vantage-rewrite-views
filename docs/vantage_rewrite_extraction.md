# VANTAGE Rewrite Extraction

This document specifies the prompt-only rewrite extraction used by SafeRoute and
Rewrite-View Lookup. The implementation lives in `asts/code_proposers.py` as
`extract_explicit_rewrites(prompt)` and `apply_boundary_rewrites(text, mapping)`.

## Scope

Rewrite extraction is deliberately conservative. It extracts only explicit
rewrite pairs visible in the user prompt before the first fenced code block.
It does not read benchmark targets, gold outputs, synthetic labels, manifest
metadata, or model outputs.

Supported rewrite syntaxes:

- `rename OLD to NEW`
- `replace OLD with NEW`
- `change OLD to NEW`
- `OLD -> NEW`, `OLD => NEW`, and `OLD → NEW`

Supported term classes:

- Identifier renames: `user -> account`
- Dotted-field substitutions: `user.name -> account.display_name`
- Leading attribute substitutions: `.name -> .display_name`
- Numeric literals: `30 -> 60`
- Simple quoted or backticked literals up to 80 characters:
  `` `pending state` -> `complete state` ``

Identifier-style substitutions are not inferred from natural language. They
are supported only when the prompt gives an explicit rewrite map, such as
`snake_case -> camelCase` or a concrete literal/token replacement. General
instructions like “make this more idiomatic” do not create a rewrite map.

## Where Extraction Looks

The extractor scans only the instruction region before the first fenced code
block. This prevents examples or comments inside the reference program from
becoming route inputs.
Rewrite-looking examples in the instruction prose before the first fenced block
are not distinguished from actual instructions; prompts used for
VANTAGE/SafeRoute should put examples inside a fenced block or avoid rewrite
syntax in non-instructional examples.

Example:

````text
Rename user to account.

```python
# rename foo to bar
return user.name
```
````

Extracted map: `{"user": "account"}`. The comment inside the fenced reference
is ignored for extraction.

## Where Replacement Applies

`apply_boundary_rewrites` applies the extracted map to the prompt-visible
reference text before tokenization. It applies across the whole reference text,
including identifiers, attributes, comments, and strings. This is intentional:
Rewrite-View Lookup builds a target-side lookup view, not a semantic Python refactor.
Safety comes from target verification, and SafeRoute backs off when the
transformed reference is unchanged.
Replacement is not applied to arbitrary examples elsewhere in the prompt; only
the extracted reference view is transformed for lookup.

## Boundary-Aware Replacement

The replacement is boundary-aware:

- Identifier replacements match complete identifier tokens.
  `user -> account` rewrites `user` and `other.user`, but not `user_id` or
  `get_user`.
- Dotted replacements match complete dotted segments.
  `client.chat -> responses.create` rewrites `client.chat(...)` and
  `old.client.chat(...)` but not `client.chatty`.
- Leading attribute replacements match attribute names with a right boundary.
  `.name -> .display_name` rewrites `user.name`, but not `user.name_extra`.
- Quoted literals that are not identifier-like use exact substring replacement.

Overlapping rewrite maps are applied longest-old-term first. This makes
`user.name -> account.display_name` win before `user -> account`.

## Positive Examples

```text
rename user to account
```

Extracted: `{"user": "account"}`

```text
replace .name with .display_name
```

Extracted: `{".name": ".display_name"}`

```text
Explicit rewrite map: user -> account, .name -> .display_name.
```

Extracted: `{"user": "account", ".name": ".display_name"}`

```text
replace `pending state` with `complete state`
```

Extracted: `{"pending state": "complete state"}`

## Negative Examples

```text
do not rename user to account
```

Extracted: `{}`

```text
rename user to user
```

Extracted: `{}`

````text
No rewrite requested.

```python
# rename user to account
return user.name
```
````

Extracted: `{}`

## Known Limitations

- This is a surface rewrite extractor, not a parser or refactoring engine.
- It does not infer broad style transforms without explicit pairs.
- It scans only before the first fenced code block. Rewrite instructions placed
  after the reference block are intentionally missed and route to exact PLD.
- It does not inspect AST scope; if the map is effective, every boundary-valid
  occurrence in the reference view is rewritten.
- It may miss complex natural-language edit instructions that do not use the
  supported forms. That miss routes to exact PLD, which is the intended safe
  fallback.
- The current artifact has unit coverage for supported forms, negated forms,
  fenced-code false positives, and boundary-aware replacement, but it does not
  include a real-prompt precision/recall benchmark. Such a benchmark is required
  before claiming production extraction reliability.
