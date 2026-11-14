# Memory probe (v1)

You answer questions about historical stock prices from memory. This is a test of what you remember.

## Input
A JSON object with `questions`: a list of `{"id": <int>, "ticker": "<ticker>", "company": "<name>", "date": "<YYYY-MM-DD>"}`.

## Task
For each question give the closing share price, in US dollars, of that stock on that date, as it was quoted on that day (not adjusted for later stock splits or dividends).

## Rules
- Give your best estimate even if you are not sure.
- Use `null` only if you have no idea at all, for example because the date is after your training data ends.
- Return exactly one answer per question id.
- Answer with compact JSON only. No other text.

## Output format
{"answers": [{"id": <int>, "close": <number or null>}]}
