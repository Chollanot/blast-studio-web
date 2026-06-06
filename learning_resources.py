"""Reference material used by Learn mode and PT mode."""
from __future__ import annotations

# Gardner grading reference - short, human-readable explanations
GARDNER_REFERENCE = {
    'expansion': {
        '2': 'Blastocyst - blastocoel cavity fills <50% of embryo',
        '3': 'Full blastocyst - cavity completely fills embryo',
        '4': 'Expanded blastocyst - cavity larger than original embryo, zona thinning',
        '5': 'Hatching blastocyst - trophectoderm has begun to herniate through the zona',
        '6': 'Hatched blastocyst - completely escaped from the zona',
    },
    'ICM': {
        'A': 'Tightly packed, many cells - excellent inner cell mass',
        'B': 'Loosely grouped, several cells - acceptable',
        'C': 'Very few cells - poor inner cell mass',
    },
    'TE': {
        'A': 'Many cells forming a cohesive epithelium - excellent trophectoderm',
        'B': 'Few cells forming a loose epithelium - acceptable',
        'C': 'Very few large cells - poor trophectoderm',
    },
}

CLINICAL_ACTION_REFERENCE = {
    'P1':  ('High Priority',
            'Top transfer candidate. Morphology consistent with the best '
            'predicted implantation potential in the cohort.'),
    'LP':  ('Low Potential',
            'Embryo with limited developmental potential. Transfer only '
            'if no better candidate is available.'),
    'REV': ('Review',
            'The morphology does not fit a fixed Gardner template; the '
            'embryologist should defer to the attending physician for a '
            'final decision.'),
}

# Transfer-priority ordering (best -> worst) used by Cohort mode
TRANSFER_PRIORITY = [
    # Top tier
    'P1', '5AA', '4AA', '3AA',
    # Good
    '5AB', '4AB', '3AB', '5BB', '4BB', '3BB', '6BB',
    # Fair
    '2AB', '2BB', '5AC', '4AC', '3AC', '5BC', '4BC', '3BC',
    # Low potential
    '2BC', '3CC', '4CC', 'LP',
    # Needs review
    'REV',
]


def grade_explanation(grade: str) -> str:
    """Plain-English breakdown of any 24-class grade."""
    if grade in CLINICAL_ACTION_REFERENCE:
        title, desc = CLINICAL_ACTION_REFERENCE[grade]
        return f"**{grade} - {title}.** {desc}"
    if len(grade) == 3 and grade[0].isdigit():
        e, i, t = grade[0], grade[1], grade[2]
        return (f"**{grade}.** "
                f"Expansion {e}: {GARDNER_REFERENCE['expansion'].get(e,'?')}. "
                f"ICM {i}: {GARDNER_REFERENCE['ICM'].get(i,'?')}. "
                f"TE {t}: {GARDNER_REFERENCE['TE'].get(t,'?')}.")
    return f"**{grade}.** (no reference available)"


def transfer_rank(grade: str) -> int:
    """Lower number = higher transfer priority. Unknown grades get last rank."""
    try:    return TRANSFER_PRIORITY.index(grade)
    except: return len(TRANSFER_PRIORITY)


# Friendly difficulty rating for PT mode
def difficulty_of(grade: str) -> str:
    easy = {'5AA','4AA','3AA','6BB','P1','LP','REV','2AB','2BB'}
    hard = {'3AB','3AC','4AC','5AC','4AB','5AB','3BB','4BB','5BB'}
    if grade in easy: return 'Easy'
    if grade in hard: return 'Hard'
    return 'Medium'
