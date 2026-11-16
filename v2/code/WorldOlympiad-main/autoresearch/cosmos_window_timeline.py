"""Frame-exact publication of closed-loop predicted windows at observed feedback times."""
def publication_counts(windows, frame_count):
    if not windows:
        raise ValueError("No recorded feedback windows")
    cursor = 0
    counts = []
    for index, record in enumerate(windows):
        if record["index"] != index or record["conditioning_frame"] != cursor:
            raise ValueError("Non-consecutive feedback timeline")
        count = record["executed_count"]
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 32:
            raise ValueError("Invalid number of executed native actions")
        counts.append(count)
        cursor += count
    if cursor + 1 != frame_count:
        raise ValueError("Published frame count differs from executed transitions")
    counts[-1] += 1
    return counts
