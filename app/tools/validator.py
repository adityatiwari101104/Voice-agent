def validate_marks(total_marks, sections):
    calculated = sum(
        section["count"] * section["marks_each"]
        for section in sections
    )

    return {
        "valid": calculated == total_marks,
        "requested_marks": total_marks,
        "calculated_marks": calculated,
        "difference": total_marks - calculated
    }