"""Very simple safety layer: reject generations containing self-harm / violence terms and regenerate.
A real system would use a trained safety classifier; this keeps the demo transparent."""
import re
BLOCK = re.compile(r"\b(kill|suicid\w*|die|dead|self[- ]?harm|cut myself|hurt myself|weapon|gun|bitch\w*|fuck\w*|shit\w*|damn)\b", re.I)  # risk + profanity terms
def is_safe(text: str) -> bool:
    return not BLOCK.search(text)
