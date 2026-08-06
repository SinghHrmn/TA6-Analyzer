"""
Realistic answer + free-text content generation for synthetic TA6 forms.
==========================================================================
Real completed TA6 forms are 90-99% filled with mostly favourable/negative
(low-risk) answers and only occasional genuine "Yes" disclosures -- sellers
answer honestly, but most disclosure questions ("are you aware of any
dispute/flood/claim/...") are legitimately "No" for most properties most of
the time. The v1 generator ignored this and used near-uniform random noise
(and only covered 13/147 questions). This module:

  - picks a per-question Yes probability from keyword heuristics tuned to
    match how real UK TA6 disclosure answers skew (see `YES_PROB_RULES`),
  - writes genuine-sounding free text for "if yes, give details" fields
    instead of a fixed lorem-ipsum pool,
  - keeps the fault rate LOW (few missing / few wrong per form, not ~12%
    of answers) so bulk realism matches real practice.
"""
import random
import re

# ----------------------------------------------------------------------
# Identity / header value pools
# ----------------------------------------------------------------------
STREETS = ["Elm Close", "Hazel Grove", "Priory Walk", "Marlow Rise", "Kestrel Way",
           "Weavers Lane", "Ashford Terrace", "Beckett Mews", "Sandpiper Drive",
           "Orchard Way", "Foxglove Road", "Mill Bank", "Cherry Tree Avenue"]
TOWNS = [("Northgate", "NG"), ("Bramfield", "BR"), ("Westbourne", "WB"),
         ("Cavendish", "CV"), ("Harlestone", "HR"), ("Fenwick", "FN")]
FIRST = ["James", "Aisha", "Robert", "Priya", "Daniel", "Sofia", "Michael", "Grace",
         "Owen", "Leah", "Thomas", "Olivia", "William", "Chloe", "Henry", "Amara"]
LAST = ["Whitmore", "Okafor", "Bennett", "Sharma", "Duncan", "Rossi", "Hartley",
        "Nguyen", "Fitzgerald", "Adeyemi", "Kowalski", "Osei"]
SOLICITOR_FIRMS = ["Marchmont & Reeve LLP", "Thornbury Legal", "Caldwell Stone Solicitors",
                    "Ashfield Conveyancing", "Priory Gate Law", "Redmayne Fisher LLP"]
ENERGY_PROVIDERS = ["British Gas", "Octopus Energy", "E.ON Next", "EDF Energy", "OVO Energy", "SSE"]
WATER_PROVIDERS = {"NG": "Northgate Water", "BR": "Anglian Water", "WB": "Thames Water",
                    "CV": "Severn Trent", "HR": "Yorkshire Water", "FN": "United Utilities"}
ELECTRICIANS = ["Sparkwell Electrical Ltd", "BrightCurrent Electrical", "SafeCircuit Contractors"]
BUILDERS = ["Redstone Builders Ltd", "Oakframe Construction", "Millbrook Building Services"]
WARRANTY_PROVIDERS = ["NHBC", "Premier Guarantee", "LABC Warranty", "Checkmate Warranties"]


def rng_seed(rng):
    return rng


def fake_person(rng):
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def fake_address(rng):
    town, code = rng.choice(TOWNS)
    return {
        "line1": f"{rng.randint(1, 180)} {rng.choice(STREETS)}",
        "town": town, "code": code,
        "postcode": f"{code}{rng.randint(1,9)} {rng.randint(1,9)}{rng.choice('ABDEFHJLNPQRSTUWXYZ')}{rng.choice('ABDEFHJLNPQRSTUWXYZ')}",
    }


def fake_date(rng, start_year=2015, end_year=2025):
    y = rng.randint(start_year, end_year)
    m = rng.randint(1, 12)
    return f"{m:02d}/{y}"


def fake_year(rng, start_year=2000, end_year=2024):
    return str(rng.randint(start_year, end_year))


# ----------------------------------------------------------------------
# Yes-probability heuristics, tuned to realistic UK disclosure-answer skew.
# Matched against the (lower-cased) question prompt text. First match wins;
# order matters -- more specific patterns first.
# ----------------------------------------------------------------------
YES_PROB_RULES = [
    # near-universal "yes" (ordinary factual / connection questions)
    (r"do you insure the property", 0.97),
    (r"do you live at the property", 0.88),
    (r"mains (electricity|gas|water)", 0.94),
    (r"mains sewerage", 0.90),
    (r"boiler and heating system working", 0.93),
    (r"reasonable care will be taken", 0.97),
    (r"clean and tidy condition", 0.96),
    (r"sufficient to pay off.*mortgages", 0.92),
    (r"certificates for electrical installation", 0.55),
    (r"electrical installation condition report", 0.5),
    (r"used only to provide hot water or heating", 0.8),
    (r"own the system outright", 0.85),
    # moderate / genuinely mixed
    (r"extension|loft conversion|garage conversion|conservatory|removal of internal walls|"
     r"removal of chimney breast|insulation|glazed doors|other building works", 0.3),
    (r"has all this work been completed", 0.9),
    (r"competent person certificates", 0.7),
    (r"new home warranty|damp proofing|timber treatment|windows, roof lights|roofing|"
     r"boiler or heating systems|underpinning", 0.22),
    (r"electric vehicle .?ev.? charging point", 0.18),
    (r"replacement to the heating system", 0.25),
    (r"solar|photovoltaic|generating electricity, hot water or heating", 0.12),
    (r"telephone", 0.55), (r"broadband", 0.85),
    (r"septic tank|sewage treatment plant|cesspool", 0.1),
    (r"shared.*heat pump", 0.05),
    (r"foul water drainage", 0.9), (r"surface water drainage", 0.75),
    (r"special requirements about a moving date", 0.2),
    (r"depend on you completing the purchase", 0.25),
    (r"anyone else, aged 17 or over, live", 0.22),
    (r"are your tenants or have any right to occupy", 0.08),
    # low-probability "yes" (disclosure of a problem -- most properties are clean)
    (r"dispute|complaint|disagreement", 0.07),
    (r"notice|proposal.*(develop|alter|change the use)", 0.06),
    (r"flood", 0.05),
    (r"japanese knotweed", 0.03),
    (r"radon", 0.04),
    (r"defences installed", 0.03),
    (r"green deal", 0.02),
    (r"claims under any of these guarantees|breach the terms", 0.03),
    (r"insurance.*difficult to obtain|buildings insurance claims", 0.05),
    (r"listed\??$", 0.04), (r"conservation area", 0.15),
    (r"tree preservation|works carried out on those trees", 0.06),
    (r"unfinished work or work that does not have all necessary consents", 0.03),
    (r"planning or building control issues", 0.03),
    (r"asked to contribute towards the cost", 0.12),
    (r"exercise any rights or arrangements over any other propert", 0.15),
    (r"other propert.*exercise any rights", 0.1),
    (r"drains, pipes or wires serving the property that cross", 0.15),
    (r"drains, pipes or wires leading to any other property", 0.1),
    (r"agreement or arrangement about drains", 0.1),
    (r"permit.*on-road parking", 0.3),
    (r"cross the public pavement", 0.4),
    (r"electrical installation works carried out", 0.35),
    (r"sewerage system discharge to the ground", 0.1),
    (r"discharges to the ground.*independent", 0.5),
    (r"shared with other properties", 0.15),
    (r"outside the boundar", 0.1),
]
DEFAULT_YES_PROB = 0.15


def yes_probability(prompt: str) -> float:
    p = (prompt or "").lower()
    for pat, prob in YES_PROB_RULES:
        if re.search(pat, p):
            return prob
    return DEFAULT_YES_PROB


NK_PROB = 0.04                # baseline chance of "Not known" where offered
FAULT_PROB_MISSING_DETAIL = 0.02   # chance a "Yes" answer's detail box is left blank
FAULT_PROB_UNANSWERED = 0.004      # chance a yes/no question itself is left blank
# Tuned so a 147-question form has ~0.5-1.5 labelled faults on average (a few,
# not a fault on every page) -- matching "few errors and few missing things".


# ----------------------------------------------------------------------
# Free-text detail generation, chosen by keyword match against the prompt.
# ----------------------------------------------------------------------
def detail_text(prompt: str, rng: random.Random, seller_name: str = "") -> str:
    p = (prompt or "").lower()
    yr = fake_year(rng)
    if re.search(r"dispute|complaint|disagreement", p):
        return rng.choice([
            f"A boundary fence disagreement with the neighbouring property was raised in {yr}; "
            "resolved informally by agreement between the parties, no further action taken.",
            f"A noise complaint was made to the local council in {yr} regarding a neighbouring "
            "property; the matter was resolved and no ongoing dispute exists.",
        ])
    if re.search(r"notice|proposal.*(develop|alter|change the use)", p):
        return (f"Notice was received from the local authority in {yr} regarding proposed "
                "highway works nearby; no direct impact on this property.")
    if "flood" in p:
        return (f"Minor surface water pooling in the rear garden after heavy rainfall in {yr}; "
                "the property itself was not internally affected.")
    if "knotweed" in p:
        return (f"A small stand of Japanese knotweed was identified near the rear boundary in {yr} "
                f"and treated by a specialist contractor under a {rng.randint(5,10)}-year insurance-backed guarantee.")
    if "radon" in p:
        return f"A radon test was carried out in {yr}; result was within the recommended action level."
    if re.search(r"extension|loft conversion|garage conversion|conservatory|glazed doors|"
                 r"removal of internal walls|removal of chimney breast|insulation|other building works", p):
        yr2 = fake_year(rng, 2008, 2023)
        return rng.choice([
            f"Single-storey rear extension completed in {yr2} by {rng.choice(BUILDERS)}; "
            "building regulations completion certificate held.",
            f"Loft conversion completed in {yr2} by {rng.choice(BUILDERS)}; signed off under the "
            "competent person scheme.",
            f"Cavity wall and loft insulation upgraded in {yr2} by a certified installer.",
        ])
    if re.search(r"new home warranty|damp proofing|timber treatment|windows|roofing|"
                 r"boiler or heating systems|underpinning", p):
        return (f"{rng.choice(WARRANTY_PROVIDERS)} guarantee taken out in {fake_year(rng,2010,2022)}, "
                f"transferable to the buyer; {rng.randint(2,10)} years remaining.")
    if "claims under any of these guarantees" in p or "breach the terms" in p:
        return f"A minor claim was made in {yr} for a leak covered under the roofing guarantee; resolved."
    if "insurance" in p and ("difficult to obtain" in p or "special condition" in p):
        return (f"Premium increased in {yr} following a regional flood-risk reassessment; "
                "cover has continued without a break.")
    if "buildings insurance claims" in p:
        return f"A claim was made in {yr} for storm damage to guttering; settled by the insurer."
    if "electrical installation" in p or "eicr" in p:
        return (f"Full rewire completed in {fake_year(rng,2012,2022)} by {rng.choice(ELECTRICIANS)}; "
                "EICR certificate rated satisfactory.")
    if "solar" in p or "photovoltaic" in p or "generating electricity" in p:
        return f"Solar PV system installed in {fake_year(rng,2015,2023)} under the MCS scheme."
    if "tree" in p or "preservation" in p:
        return f"Crown reduction works carried out on a rear-garden tree in {yr} with local authority consent."
    if "occupiers" in p or "aged 17" in p:
        return f"{seller_name or fake_person(rng)}'s adult child resides at the property and will vacate on completion."
    if "another property" in p or "moving date" in p:
        return "Purchase is linked to a related onward transaction; a simultaneous exchange and completion is preferred."
    if "parking" in p:
        return "Off-street parking on a private driveway to the front of the property; no permit required."
    if "boundary" in p:
        return "Boundary line agreed informally with the neighbour; no formal transfer of ownership has taken place."
    if "drain" in p or "pipe" in p or "wire" in p:
        return "A shared drainage run crosses the rear of the neighbouring property under a longstanding informal arrangement."
    return "Further details available on request; documentation held by the seller's solicitor."


def role_label():
    return "Seller"


TELECOM_PROVIDERS = ["BT", "Virgin Media", "Sky", "TalkTalk", "Vodafone"]


def structured_value(label: str, rng: random.Random, context: str = ""):
    """For detail/orphan boxes that want a short STRUCTURED value (a provider
    name, a meter location, a reference number, a date) rather than a
    narrative sentence -- decided from the box's OWN printed caption, not the
    parent question's prompt. `context` (the parent question's prompt, if
    known) disambiguates a generic "Provider's name:" caption between an
    energy company, a water company, or a telecoms company. Returns None if
    the label doesn't match a known structured pattern (caller should fall
    back to a narrative sentence)."""
    lab = (label or "").lower()
    ctx = (context or "").lower()
    if "provider" in lab:
        if re.search(r"water|sewerage|sewage", ctx):
            return rng.choice(list(WATER_PROVIDERS.values()))
        if re.search(r"telephone|broadband", ctx):
            return rng.choice(TELECOM_PROVIDERS)
        return rng.choice(ENERGY_PROVIDERS)
    if "location of meter" in lab or "location of stopcock" in lab:
        return rng.choice(["Under stairs cupboard", "External meter box, front elevation",
                           "Utility room", "Kitchen cupboard"])
    if "mpan" in lab:
        return "".join(str(rng.randint(0, 9)) for _ in range(13))
    if "mprn" in lab:
        return "".join(str(rng.randint(0, 9)) for _ in range(6))
    if "make/model" in lab:
        return rng.choice(["Worcester Bosch Greenstar 30i", "Vaillant ecoTEC Plus 832", "Ideal Logic Max 30"])
    if "who insures the property" in lab:
        return rng.choice(["Freeholder's block policy (managing agent-arranged)",
                           "Landlord's buildings policy"])
    if re.search(r"\byear\b", lab) and "month" not in lab:
        return fake_year(rng)
    if "month/year" in lab or (lab.strip() == "date:"):
        return fake_date(rng)
    return None
