"""
Name database for AI keyboard predictions
Contains common first names and surnames for better prediction accuracy
"""

# Top US Census first names (both male and female)
FIRST_NAMES = [
    # Male names
    "JAMES", "ROBERT", "JOHN", "MICHAEL", "DAVID", "WILLIAM", "RICHARD", "CHARLES",
    "JOSEPH", "THOMAS", "CHRISTOPHER", "DANIEL", "PAUL", "MARK", "DONALD", "STEVEN",
    "KENNETH", "JOSHUA", "KEVIN", "BRIAN", "GEORGE", "TIMOTHY", "RONALD", "JASON",
    "EDWARD", "JEFFREY", "RYAN", "JACOB", "GARY", "NICHOLAS", "ERIC", "JONATHAN",
    "STEPHEN", "LARRY", "JUSTIN", "SCOTT", "BRANDON", "BENJAMIN", "SAMUEL", "FRANK",
    "MATTHEW", "GREGORY", "RAYMOND", "ALEXANDER", "PATRICK", "JACK", "DENNIS", "JERRY",
    
    # Female names
    "MARY", "PATRICIA", "JENNIFER", "LINDA", "ELIZABETH", "BARBARA", "SUSAN", "JESSICA",
    "SARAH", "KAREN", "NANCY", "LISA", "BETTY", "HELEN", "SANDRA", "DONNA", "CAROL",
    "RUTH", "SHARON", "MICHELLE", "LAURA", "SARAH", "KIMBERLY", "DEBORAH", "DOROTHY",
    "AMY", "ANGELA", "ASHLEY", "BRENDA", "EMMA", "OLIVIA", "CYNTHIA", "MARIE", "JANET",
    "CATHERINE", "FRANCES", "CHRISTINE", "SAMANTHA", "DEBRA", "RACHEL", "CAROLYN",
    "JANET", "VIRGINIA", "MARIA", "HEATHER", "DIANE", "JULIE", "JOYCE", "VICTORIA",
    
    # Additional popular names
    "NOAH", "LIAM", "ETHAN", "MASON", "LOGAN", "LUCAS", "HENRY", "OWEN", "CALEB",
    "SOPHIA", "ISABELLA", "CHARLOTTE", "AMELIA", "MIA", "HARPER", "EVELYN", "ABIGAIL",
    "EMILY", "ELLA", "ELIZABETH", "CAMILA", "LUNA", "SOFIA", "AVERY", "MILA", "ARIA"
]

# Common surnames from various sources
SURNAMES = [
    "SMITH", "JOHNSON", "WILLIAMS", "BROWN", "JONES", "GARCIA", "MILLER", "DAVIS",
    "RODRIGUEZ", "MARTINEZ", "HERNANDEZ", "LOPEZ", "GONZALEZ", "WILSON", "ANDERSON",
    "THOMAS", "TAYLOR", "MOORE", "JACKSON", "MARTIN", "LEE", "PEREZ", "THOMPSON",
    "WHITE", "HARRIS", "SANCHEZ", "CLARK", "RAMIREZ", "LEWIS", "ROBINSON", "WALKER",
    "YOUNG", "ALLEN", "KING", "WRIGHT", "SCOTT", "TORRES", "NGUYEN", "HILL", "FLORES",
    "GREEN", "ADAMS", "NELSON", "BAKER", "HALL", "RIVERA", "CAMPBELL", "MITCHELL",
    "CARTER", "ROBERTS", "GOMEZ", "PHILLIPS", "EVANS", "TURNER", "DIAZ", "PARKER",
    "CRUZ", "EDWARDS", "COLLINS", "REYES", "STEWART", "MORRIS", "MORALES", "MURPHY",
    "COOK", "ROGERS", "GUTIERREZ", "ORTIZ", "MORGAN", "COOPER", "PETERSON", "BAILEY",
    "REED", "KELLY", "HOWARD", "RAMOS", "KIM", "COX", "WARD", "RICHARDSON", "WATSON",
    "BROOKS", "CHAVEZ", "WOOD", "JAMES", "BENNETT", "GRAY", "MENDOZA", "RUIZ", "HUGHES"
]

def get_all_names():
    """Return all names (first names + surnames) as a set"""
    return set(FIRST_NAMES + SURNAMES)

def get_names_for_sequence(button_sequence, groups):
    """
    Get names that match a specific button sequence
    
    Args:
        button_sequence: List of button presses
        groups: Dictionary mapping buttons to letter groups
    
    Returns:
        List of matching names
    """
    all_names = get_all_names()
    matching_names = []
    
    for name in all_names:
        if len(name) == len(button_sequence):
            match = True
            for i, letter in enumerate(name):
                button = button_sequence[i]
                if letter not in groups.get(button, ""):
                    match = False
                    break
            if match:
                matching_names.append(name)
    
    return matching_names

def is_name(word):
    """Check if a word is in our name database"""
    return word.upper() in get_all_names()