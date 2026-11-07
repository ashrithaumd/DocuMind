"""Builds the long documents of the Phase 3 corpus (kb/docs/), reproducibly.

The short documents in kb/docs/ are hand-written. These long ones
(handbooks, service manuals, contracts, postmortems, release notes) are
assembled here from hand-written section text and generated reference
tables, so the corpus reaches several hundred chunks without anyone typing
100,000 words. A fixed seed makes every run produce the same files:

    python kb/tools/build_long_docs.py        (from the repo root)

What it plants on purpose, for Phase 6's eval questions:
  - exact-term facts: error codes, part numbers, ticket ids, config keys,
    clause numbers. Full-text search should find these; embeddings blur them.
  - paraphrase facts: written in words a question would not use (e.g.
    "step away on full pay to look after a relative" for caregiver leave).
    Dense search should find these.
  - versioned near-duplicates: 2025 and 2026 handbooks with changed numbers,
    two manuals with the same structure for different products. Each states
    its year and version in its text.
  - mirrored facts across the two orgs: the same topics, different numbers,
    so a cross-org leak shows up as a wrong answer.
A ledger of the planted facts is written to kb/facts.yaml.
"""
import random
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SEED = 20261107

LEDGER: list[dict] = []


def fact(doc: str, kind: str, text: str, ask: str) -> None:
    """Record a planted fact: kind is exact | paraphrase | versioned | mirrored."""
    LEDGER.append({"doc": doc, "kind": kind, "fact": text, "ask": ask})


def block(text: str, **params) -> str:
    return dedent(text).strip().format(**params)


# =============================================================================
# Organisations and the numbers that differ between them (mirrored facts)
# =============================================================================

ORGS = {
    "aurora": {
        "company": "Aurora Robotics", "short": "Aurora", "domain": "aurora-robotics.example",
        "hq": "Pittsburgh", "hr_tool": "PeopleDesk", "chat": "Slack", "wiki": "Notion",
        "people_team": "the People team", "ceo": "Ingrid Halvorsen", "it_desk": "it@aurora-robotics.example",
        "office": "Building C on Smallman Street", "benefits_vendor": "Keystone Health",
    },
    "bluefin": {
        "company": "Bluefin Analytics", "short": "Bluefin", "domain": "bluefin.example",
        "hq": "Boston (registered office; we are remote-first)", "hr_tool": "Rippling", "chat": "Slack",
        "wiki": "Confluence", "people_team": "People Operations", "ceo": "Marcus Achebe",
        "it_desk": "it-help@bluefin.example", "office": "our coworking hubs in Boston, Toronto and Porto",
        "benefits_vendor": "Harborline Benefits",
    },
}

# Handbook numbers per org and year. 2025 -> 2026 changes are the versioned
# distractors; aurora vs bluefin differences are the mirrored ones.
HANDBOOK = {
    ("aurora", 2025): dict(version="2025.1", effective="1 January 2025", vacation=20, sick=8,
                           parental=14, caregiver=8, bereavement=3, match=4, vest_months=0,
                           learning=1500, wellness=40, remote_days=2, review="March and September",
                           notice_weeks=2, meal=65, hotel=200, hotel_big=280, mileage="0.58",
                           claim_days=45, sabbatical_years=6, sabbatical_weeks=4, jury_days=5,
                           referral=2500, phone=40, volunteer_days=1, carryover=5),
    ("aurora", 2026): dict(version="2026.1", effective="1 March 2026", vacation=22, sick=10,
                           parental=16, caregiver=10, bereavement=5, match=5, vest_months=0,
                           learning=2000, wellness=60, remote_days=3, review="April and October",
                           notice_weeks=2, meal=75, hotel=220, hotel_big=310, mileage="0.62",
                           claim_days=30, sabbatical_years=5, sabbatical_weeks=4, jury_days=10,
                           referral=3000, phone=50, volunteer_days=2, carryover=5),
    ("bluefin", 2025): dict(version="2025.2", effective="1 February 2025", vacation=25, sick=10,
                            parental=18, caregiver=12, bereavement=5, match=4, vest_months=6,
                            learning=1800, wellness=50, remote_days=5, review="January and July",
                            notice_weeks=3, meal=55, hotel=175, hotel_big=240, mileage="0.60",
                            claim_days=60, sabbatical_years=5, sabbatical_weeks=6, jury_days=10,
                            referral=3500, phone=45, volunteer_days=2, carryover=8),
    ("bluefin", 2026): dict(version="2026.1", effective="1 January 2026", vacation=27, sick=12,
                            parental=20, caregiver=14, bereavement=7, match=5, vest_months=0,
                            learning=2500, wellness=75, remote_days=5, review="February and August",
                            notice_weeks=3, meal=60, hotel=190, hotel_big=260, mileage="0.62",
                            claim_days=45, sabbatical_years=4, sabbatical_weeks=6, jury_days=15,
                            referral=4000, phone=50, volunteer_days=3, carryover=10),
}


# =============================================================================
# Employee handbooks (4 documents: 2 orgs x 2 years)
# =============================================================================

HANDBOOK_SECTIONS = [
    ("About this handbook", """
     This handbook describes how we work at {company}: the rules everyone follows, the benefits
     you can count on, and where to go when something is unclear. It is version {version} of the
     handbook, written for the year {year} and effective from {effective}. It replaces every
     earlier version. Where this handbook and an older team page disagree, this handbook wins;
     where it and the law of the place you work disagree, the law wins and {people_team} will
     tell you how.

     We review the handbook once a year. Suggestions are welcome at any time: open a request in
     {hr_tool} or write to {people_team}. Small corrections are published straight away and
     listed in the change log at the end; anything that changes a benefit or an entitlement waits
     for the next yearly version, so nobody loses something they planned around in the middle of
     the year.

     A note on tone. We tried to write this the way we would explain it to a new colleague over
     coffee, not the way a lawyer would. When a sentence matters legally, it says so plainly.
     """),
    ("Our working hours", """
     Most of us work about forty hours a week. We do not track hours for salaried staff, and we
     do not reward people for being online late. What we ask is that you are reachable during your
     team's core hours, that you keep your calendar honest, and that you tell your team when you
     will be away.

     Teams agree their own core hours and publish them on their team page. Meetings outside core
     hours need a good reason and should rotate, so the same people are not always the ones on
     an early or late call.

     Under this version of the handbook, people whose role allows it may work away from the office
     for up to {remote_days} days a week. Field and lab roles that need hands on hardware agree a
     pattern with their manager instead. Working from another country is covered under "Travel
     and working abroad" below.
     """),
    ("Paid time off", """
     Every full-time employee receives {vacation} days of paid vacation each calendar year, on top
     of public holidays. Part-time colleagues receive the same number pro rata. Vacation builds up
     evenly through the year, but you may take days before you have built them up, as long as your
     manager agrees.

     You may carry up to {carryover} unused vacation days into the next year. Carried-over days
     must be used by 31 March or they lapse; we would rather you rested than banked days.

     Please give at least two weeks' notice for any absence of more than three days, and record
     every day off in {hr_tool} so payroll and your team both know. Managers should answer a
     request within three working days. A request that has not been answered in that time may be
     escalated to {people_team}.
     """),
    ("When you are unwell", """
     If you are ill, stay home and get better. You have {sick} paid sick days a year, and you do
     not need a doctor's note for absences of up to three consecutive days. For longer absences,
     {people_team} will ask for a note and will check in with you about what support you need.

     Tell your manager as early as you can on the first day, by message or phone, and log the day
     in {hr_tool} when you are able to. Sick days are separate from vacation: if you fall ill while
     on vacation, tell us and the days you were ill are given back to you as vacation.

     For longer illnesses, our disability cover through {benefits_vendor} pays 60 percent of your
     salary after the sick days run out, for up to 26 weeks.
     """),
    ("Family leave", """
     A parent who has a new child, by birth, adoption or surrogacy, can take {parental} weeks of
     leave on full pay, at any time in the first year. It can be taken in one block or in up to
     three separate blocks. Both parents receive the same entitlement, whatever their gender.

     Colleagues who become the main carer for a close relative with a serious, long-lasting illness
     can step away from work on full pay for up to {caregiver} weeks in any twelve-month period.
     The weeks do not have to be taken together, and your role is kept open for you.

     When someone close to you dies, take the time you need. We give {bereavement} paid days of
     bereavement leave as a minimum, and your manager can agree more.
     """),
    ("Pay and payroll", """
     Salaries are paid monthly, on the last working day of the month, into the bank account you
     registered in {hr_tool}. Your payslip appears in {hr_tool} two days before payday. If a
     payment is wrong, tell {people_team} straight away; underpayments are corrected within five
     working days, outside the normal payroll run if needed.

     Salary bands are reviewed once a year. Changes from the yearly review take effect on the
     first payroll after the review closes. Your manager will explain where your pay sits in your
     band and what would move it.
     """),
    ("Retirement savings", """
     We match what you put into the company retirement plan, up to {match} percent of your base
     salary. Matching contributions belong to you immediately{vest_text}. You can change how much
     you contribute at any time in {hr_tool}; changes apply from the next payroll.

     The plan is run by an outside administrator. {people_team} holds two information sessions a
     year and can arrange a free session with an independent adviser.
     """),
    ("Health and wellbeing", """
     Health, dental and vision cover is provided through {benefits_vendor}. We pay the full premium
     for employees and 80 percent for partners and children. Cover starts on your first day.

     Everyone also receives a wellbeing allowance of {wellness} US dollars a month, which can be
     spent on gym membership, sports classes, therapy, meditation apps or similar. Claim it through
     {hr_tool} with a receipt; unused allowance does not roll over.

     Our employee assistance programme offers free, confidential counselling, around the clock,
     for you and your household. Nobody at {short} sees who uses it.
     """),
    ("Learning and growth", """
     Each employee has a learning budget of {learning} US dollars a year for courses, books,
     conferences and certifications related to their work or the next role they are aiming for.
     Spending up to the budget only needs your manager's agreement; anything above it needs a
     short written case.

     Performance conversations happen twice a year, in {review}. They are written down, shared with
     you, and never a surprise: anything serious should already have come up in your regular
     one-to-ones. Promotions are decided in the same cycles.

     After {sabbatical_years} years with us, you can take a paid sabbatical of {sabbatical_weeks}
     weeks, once in every {sabbatical_years}-year period, to rest, travel or learn something new.
     """),
    ("Travel and expenses", """
     Book travel through the company portal so we can find you if something goes wrong on a trip.
     The full rules are in the expense policy; the headline numbers for {year} are a meal allowance
     of {meal} US dollars a day, a hotel cap of {hotel} US dollars a night ({hotel_big} in the most
     expensive cities), and mileage at {mileage} US dollars a mile for your own car.

     Claims must be submitted within {claim_days} days of the expense, with a receipt. We reimburse
     you in the next payroll. A company card is available for people who travel more than four
     times a year; ask {people_team}.

     Every employee also gets {phone} US dollars a month towards their mobile phone bill, paid
     automatically with salary.
     """),
    ("Travel and working abroad", """
     Short trips abroad for work are booked like any other travel. Working from another country
     while not on a work trip is different: it can create tax and immigration obligations for you
     and for us. You may work from another country for up to 30 days in a calendar year without
     special approval, as long as you tell {people_team} before you go. Anything longer needs
     approval from {people_team} and our legal adviser, in writing, before you travel.
     """),
    ("Equipment and security", """
     You will receive a laptop and the accessories you need on or before your first day. It
     belongs to {short} and must be returned when you leave. Keep it locked when you step away, and
     never let anyone else use it, including family members.

     Report a lost or stolen device to {it_desk} as soon as you notice, whatever the time. We would
     much rather hear about a lost laptop in the first hour than the first week: we can wipe it
     remotely and nobody will be blamed for reporting quickly.

     The information security policy has the full rules, including passwords, multi-factor
     authentication and how to handle customer data.
     """),
    ("Behaviour and respect", """
     We expect everyone to treat colleagues, customers and partners with respect. Harassment,
     discrimination and bullying are not tolerated, whoever is responsible and whatever their
     role. If something happens to you or you see it happen, you can raise it with your manager,
     with {people_team}, or anonymously through the ethics line run by an outside provider.

     Every concern is looked into, quickly and fairly. Nobody who raises a concern in good faith
     will suffer for it; retaliation is itself a serious breach of this handbook.
     """),
    ("Referrals", """
     The best people we hire are often people you already know. If someone you referred is hired
     and stays for six months, you receive a referral bonus of {referral} US dollars, paid with the
     next salary. Referrals for leadership roles are welcome but do not earn the bonus.
     """),
    ("Volunteering and public duties", """
     Each year you can take {volunteer_days} paid days to volunteer for a charity or community
     group of your choice. If you are called for jury duty, we pay your full salary for up to
     {jury_days} working days, and longer if the law where you work requires it.
     """),
    ("Public holidays", """
     Offices follow the public holidays of the country they are in, and the list for {year} is
     published in {wiki} each December. If you work remotely, you follow the holidays of the
     country you live and are employed in. If your role means you must work on a public holiday,
     you receive a day off in lieu, to be taken within three months.
     """),
    ("Gifts and hospitality", """
     You may accept small gifts from customers or suppliers worth up to {gift_limit} US dollars, as
     long as they are not cash and nothing is expected in return. Anything worth more must be
     declared in {hr_tool} within five working days, and usually politely returned. Meals with a
     supplier are fine if they are occasional and the conversation is about work; never accept
     hospitality from a supplier while a purchasing decision involving them is open.
     """),
    ("Conflicts of interest", """
     Tell {people_team} about anything that could make it look as if your personal interests and
     your work pull in different directions: a second job, a stake in a supplier or competitor, or
     a close relative working for one. A declared conflict is rarely a problem; an undeclared one
     always is. Declarations are reviewed by {people_team} and your manager within ten working days.
     """),
    ("Side projects and open source", """
     What you build in your own time, on your own equipment, without using {short}'s confidential
     information, belongs to you. Contributing to open-source projects is encouraged; for projects
     related to our products, check with your manager first so we do not accidentally publish
     something we have filed for patent protection.
     """),
    ("Social media and speaking", """
     You are welcome to say where you work and to share public news about {short}. Do not share
     anything about customers, unreleased products or internal numbers. Only the communications
     team speaks for the company to journalists. If you are asked to speak at a conference about
     your work, tell your manager; we will usually pay the travel from your learning budget.
     """),
    ("Personal relationships at work", """
     Relationships between colleagues are a private matter, except where one person has authority
     over the other's pay, promotion or work. In that case, the more senior person must tell
     {people_team}, in confidence, so the reporting line can be changed. Nobody will be penalised
     for disclosing a relationship.
     """),
    ("Health and safety", """
     {safety_text}
     Report every accident and every near miss within 24 hours using the safety form, even if
     nobody was hurt. Near-miss reports are how we stop the next accident; they are never used to
     blame anyone. First-aid kits and defibrillators are marked on the floor plans in {wiki}.
     """),
    ("Data handling and privacy", """
     Customer data is the most sensitive thing we hold. Only access the customer data you need for
     the task in front of you, never copy it to personal devices or personal accounts, and never
     paste it into tools that are not on the approved list in {wiki}. Employee data is handled by
     {people_team} only. If you think data has been exposed, report it within the hour, even if you
     are not sure; we would rather investigate a false alarm.
     """),
    ("Performance and promotion", """
     Each review cycle has three steps: a short self-assessment, feedback from two or three
     colleagues you choose with your manager, and a conversation with your manager. Ratings are on
     a four-point scale: "exceeds", "strong", "developing" and "not meeting". A "not meeting" rating
     always comes with a written improvement plan of at least eight weeks and regular check-ins.

     Promotions are proposed by managers and reviewed by a panel of senior colleagues from other
     teams, so that standards are the same across the company. A promotion raises your salary by at
     least {promo_raise} percent, or to the bottom of the new band if that is higher.
     """),
    ("Compensation reviews", """
     Salaries are reviewed once a year against the market and against the bands for each level.
     Our aim is to pay at or above the median of the market for every role. Merit increases are
     decided from the yearly budget, and are higher for people low in their band who are performing
     strongly. Equity is reviewed at the same time.
     """),
    ("Parking, commuting and the office", """
     {commute_text}
     The office kitchen is stocked every weekday; please clean up after yourself. Quiet rooms can be
     booked for up to two hours at a time. Pets are not allowed in the office, except assistance
     animals.
     """),
    ("Leaving {short}", """
     If you decide to leave, please give at least {notice_weeks} weeks' notice in writing to your
     manager, or the notice period in your contract if that is longer. We will pay out vacation
     days you have built up and not used. Return your equipment on or before your last day; the
     IT team will send a prepaid box if you work remotely.

     We hold a short exit conversation with everyone who leaves. It is optional, but it is one of
     the most useful sources of honest feedback we have.
     """),
]

HANDBOOK_FAQ = [
    ("Can I take vacation in my first month?",
     "Yes. Vacation builds up through the year, but you can take days early with your manager's "
     "agreement. If you leave before you have built them up, the difference is deducted from your "
     "final pay."),
    ("Do public holidays count against my vacation days?",
     "No. Public holidays are on top of your {vacation} vacation days."),
    ("My child is ill. Is that a sick day?",
     "Short absences to look after an ill child can be logged as sick days. For a long-term "
     "illness of a close relative, see family leave."),
    ("Who approves a training course?",
     "Your manager, as long as it fits in your {learning} US dollar budget for the year."),
    ("How fast are expenses reimbursed?",
     "In the next payroll after your claim is approved. Submit within {claim_days} days."),
    ("Can I work from a café or a coworking space?",
     "Yes, as long as you can take calls privately and you never leave your laptop unattended."),
    ("What happens to my retirement match if I leave?",
     "Matching contributions are yours{vest_text}, and stay in the plan or can be moved."),
    ("Can I buy extra vacation days?",
     "Not under this version of the handbook. Unpaid leave of up to four weeks can be agreed with "
     "your manager."),
    ("Who do I tell if I am going to be late or absent?",
     "Your manager, as early as you can, and your team channel in {chat}."),
    ("A supplier sent me a bottle of wine. Can I keep it?",
     "If it is worth no more than {gift_limit} US dollars and nothing is expected in return, yes. "
     "Otherwise declare it in {hr_tool} within five working days."),
    ("Can I work a second job?",
     "Usually yes, as long as it doesn't compete with us or affect your work. Declare it as a "
     "potential conflict of interest first."),
    ("How do I get a second monitor or a better chair?",
     "Ask {it_desk}. Ergonomic equipment recommended by an assessment is always approved."),
    ("When does a promotion take effect?",
     "From the first payroll after the review cycle in which it was decided, with a raise of at "
     "least {promo_raise} percent."),
    ("I lost my laptop on the train. What now?",
     "Tell {it_desk} straight away, whatever the time. Nobody is blamed for reporting quickly."),
    ("Can I bring my dog to the office?",
     "Not under this version of the handbook, except assistance animals."),
    ("Is the wellbeing allowance taxable?",
     "In most countries it is paid as a taxable benefit through payroll. {people_team} can explain "
     "the rules where you live."),
    ("How long do I have to submit a receipt?",
     "{claim_days} days from the date of the expense."),
    ("Who can see my employee assistance programme sessions?",
     "Nobody at {short}. The provider only reports anonymous totals."),
    ("Where are the latest versions of company policies?",
     "On the policies page of {wiki}. This handbook is version {version}."),
]


HANDBOOK_EXTRA = {
    "aurora": dict(
        safety_text=("Anyone who enters the robot test floor must have completed the floor safety "
                     "induction, wear safety shoes, and stay outside the yellow lines while robots "
                     "are moving. Never step into the path of a moving robot to stop it: use the "
                     "red emergency-stop buttons on every pillar."),
        commute_text=("The Pittsburgh office has 120 parking spaces, free for employees, and a "
                      "secure bike room. Public transport passes are reimbursed up to 100 US "
                      "dollars a month.")),
    "bluefin": dict(
        safety_text=("Most of us work from home. Set up your workspace safely: the home-office "
                     "stipend can pay for a proper chair, and an ergonomic assessment is available "
                     "on request. Take regular breaks away from the screen."),
        commute_text=("Our coworking hubs are open to everyone; desks are booked through the hub "
                      "app. Travel to a hub is reimbursed for team days only, up to 60 US dollars "
                      "a day.")),
}
HANDBOOK_YEAR_EXTRA = {
    ("aurora", 2025): dict(gift_limit=50, promo_raise=6),
    ("aurora", 2026): dict(gift_limit=75, promo_raise=8),
    ("bluefin", 2025): dict(gift_limit=60, promo_raise=7),
    ("bluefin", 2026): dict(gift_limit=100, promo_raise=8),
}


def handbook(org: str, year: int) -> tuple[str, str]:
    p = {**ORGS[org], **HANDBOOK[(org, year)], **HANDBOOK_EXTRA[org],
         **HANDBOOK_YEAR_EXTRA[(org, year)], "year": year}
    p["vest_text"] = (" from the day they are paid in" if not p["vest_months"]
                      else f" after {p['vest_months']} months of service")
    name = f"{org}/employee-handbook-{year}.txt"
    parts = [f"{p['company']} — Employee Handbook",
             f"Version {p['version']} — for the year {year}. Effective {p['effective']}.",
             f"Owner: {p['people_team']}. Applies to every employee of {p['company']}."]
    for n, (title, body) in enumerate(HANDBOOK_SECTIONS, start=1):
        parts.append(f"{n}. {title.format(**p)}")
        parts.append(block(body, **p))
    parts.append(f"{len(HANDBOOK_SECTIONS) + 1}. Frequently asked questions")
    for q, a in HANDBOOK_FAQ:
        parts.append(f"Q: {q}\nA: {a.format(**p)}")
    parts.append(f"{len(HANDBOOK_SECTIONS) + 2}. Change log")
    parts.append(change_log(org, year, p))
    fact(name, "paraphrase", f"Caregiver leave: up to {p['caregiver']} weeks on full pay in 12 months",
         "How much paid time off can I get to look after a family member with a long illness?")
    fact(name, "versioned", f"{year}: {p['vacation']} vacation days, {p['parental']} weeks parental leave",
         f"How many vacation days do {p['short']} employees get in {year}?")
    fact(name, "mirrored", f"Gifts up to {p['gift_limit']} USD may be accepted; more must be declared",
         "What is the most a supplier gift can be worth before I have to declare it?")
    fact(name, "mirrored", f"Wellbeing allowance {p['wellness']} USD/month; learning budget {p['learning']} USD",
         "What is the monthly wellbeing allowance?")
    return name, "\n\n".join(parts)


def change_log(org: str, year: int, p: dict) -> str:
    if year == 2025:
        return f"Version {p['version']}: yearly update. No entitlements reduced."
    old = HANDBOOK[(org, 2025)]
    changes = []
    for key, label in (("vacation", "vacation days"), ("sick", "sick days"),
                       ("parental", "weeks of parental leave"), ("caregiver", "weeks of caregiver leave"),
                       ("match", "percent retirement match"), ("learning", "US dollars learning budget"),
                       ("wellness", "US dollars monthly wellbeing allowance"),
                       ("referral", "US dollars referral bonus")):
        if old[key] != p[key]:
            changes.append(f"- {label}: {old[key]} -> {p[key]}")
    return (f"Version {p['version']} replaces version {old['version']}. Changes:\n"
            + "\n".join(changes))


# =============================================================================
# Service manuals (2 near-duplicate structures per org: same skeleton,
# different products, codes and numbers)
# =============================================================================

PRODUCTS = {
    "aurora/pika-gen3-service-manual.txt": dict(
        org="aurora", product="Pika picker, generation 3", short="Pika", version="3.2", year=2026,
        part_prefix="AUR-PK3", code_prefix="E", ticket_prefix="FOPS",
        subsystems=[("1", "drive"), ("2", "gripper"), ("3", "vision"), ("4", "battery"),
                    ("5", "lift mast"), ("6", "network")],
        specs=[("Maximum payload", "12 kg"), ("Pick rate", "up to 600 items per hour"),
               ("Shelf reach", "0.2 m to 2.4 m"), ("Battery", "6 hours picking, 55 minutes to 80 percent"),
               ("Weight", "148 kg"), ("Top speed", "1.8 m/s"), ("Ingress rating", "IP54"),
               ("Operating temperature", "0 to 40 degrees Celsius")],
        planted=("E-2417", "GRIPPER_VACUUM_LOW", "AUR-PK3-4471-B",
                 "the suction cup kit", "vacuum below 55 kPa for more than 2 seconds"),
    ),
    "aurora/atlas-service-manual.txt": dict(
        org="aurora", product="Atlas mover, generation 2", short="Atlas", version="2.4", year=2026,
        part_prefix="AUR-AT2", code_prefix="A", ticket_prefix="FOPS",
        subsystems=[("1", "drive"), ("2", "lift deck"), ("3", "vision"), ("4", "battery"),
                    ("5", "safety scanner"), ("6", "network")],
        specs=[("Maximum payload", "1,200 kg"), ("Deck lift", "60 mm in 2.5 seconds"),
               ("Battery", "48 V, 105 Ah, 9 hours at 60 percent duty"), ("Weight", "410 kg"),
               ("Top speed", "2.0 m/s unloaded, 1.2 m/s loaded"), ("Turning", "on the spot"),
               ("Ingress rating", "IP54"), ("Operating temperature", "-5 to 45 degrees Celsius")],
        planted=("A-5032", "SAFETY_FIELD_OBSTRUCTED", "AUR-AT2-7718-C",
                 "the safety scanner window", "the protective field reports an obstacle for more than 30 seconds"),
    ),
    "bluefin/current-v4-operations-manual.txt": dict(
        org="bluefin", product="Current forecasting, version 4", short="Current", version="4.6", year=2026,
        part_prefix="CUR-CFG", code_prefix="CUR", ticket_prefix="SUP",
        subsystems=[("1", "ingestion"), ("2", "feature store"), ("3", "training"), ("4", "scoring"),
                    ("5", "export"), ("6", "scheduler")],
        specs=[("Forecast horizon", "8 weeks, daily"), ("Refresh", "nightly at 02:00 Eastern"),
               ("Accuracy target", "WAPE under 28 percent at store-product-week"),
               ("History needed", "at least 2 years of point-of-sale data"),
               ("Largest tenant", "about 1.2 billion store-product-day rows"),
               ("Export formats", "daily CSV drop, REST API"), ("API rate limit", "600 requests per minute"),
               ("Data retention", "forecast history kept for 400 days")],
        fixed_codes={"PROMO_CALENDAR_INVALID": "CUR-2215"},  # as quoted in the 2026 postmortem
        planted=("CUR-4107", "SCORING_FEATURES_STALE", "CUR-CFG-scoring.max_feature_age_hours",
                 "the feature freshness limit", "the newest feature row is older than 30 hours"),
    ),
    # Older generations: the same structure and some of the same error codes
    # with different meanings or fixes, so a question must land on the right
    # version (versioned distractors).
    "aurora/pika-gen2-service-manual.txt": dict(
        org="aurora", product="Pika picker, generation 2", short="Pika gen 2", version="2.9", year=2024,
        part_prefix="AUR-PK2", code_prefix="E", ticket_prefix="FOPS",
        subsystems=[("1", "drive"), ("2", "gripper"), ("3", "vision"), ("4", "battery"),
                    ("5", "lift mast"), ("6", "network")],
        specs=[("Maximum payload", "8 kg"), ("Pick rate", "up to 450 items per hour"),
               ("Shelf reach", "0.3 m to 2.1 m"), ("Battery", "5 hours picking, 70 minutes to 80 percent"),
               ("Weight", "139 kg"), ("Top speed", "1.5 m/s"), ("Ingress rating", "IP42"),
               ("Operating temperature", "5 to 35 degrees Celsius")],
        planted=("E-2417", "GRIPPER_VACUUM_LOW", "AUR-PK2-3307-A",
                 "the vacuum pump diaphragm", "vacuum below 40 kPa for more than 5 seconds"),
    ),
    "bluefin/current-v3-operations-manual.txt": dict(
        org="bluefin", product="Current forecasting, version 3 (legacy)", short="Current v3", version="3.8",
        year=2025, part_prefix="CUR3-CFG", code_prefix="CUR", ticket_prefix="SUP",
        subsystems=[("1", "ingestion"), ("2", "feature store"), ("3", "training"), ("4", "scoring"),
                    ("5", "export"), ("6", "scheduler")],
        specs=[("Forecast horizon", "6 weeks, daily"), ("Refresh", "nightly at 03:00 Eastern"),
               ("Accuracy target", "WAPE under 32 percent at store-product-week"),
               ("History needed", "at least 3 years of point-of-sale data"),
               ("Fallback", "ARIMA model when the main model fails"),
               ("Export formats", "daily CSV drop only"), ("API rate limit", "300 requests per minute"),
               ("Data retention", "forecast history kept for 180 days")],
        fixed_codes={"PROMO_CALENDAR_INVALID": "CUR-2215"},  # codes are stable across versions
        planted=("CUR-4107", "SCORING_FEATURES_STALE", "CUR3-CFG-scoring.max_feature_age_hours",
                 "the feature freshness limit", "the newest feature row is older than 48 hours"),
    ),
    "bluefin/tide-admin-guide.txt": dict(
        org="bluefin", product="Tide dashboards", short="Tide", version="7.11", year=2026,
        part_prefix="TIDE-CFG", code_prefix="TD", ticket_prefix="SUP",
        subsystems=[("1", "connector"), ("2", "semantic layer"), ("3", "query engine"), ("4", "cache"),
                    ("5", "sharing"), ("6", "sso")],
        specs=[("Dashboards", "25 in Standard, unlimited in Enterprise"),
               ("Refresh", "every 4 hours; hourly in Enterprise"),
               ("Load time", "under 3 seconds at p90 for up to 500 million rows"),
               ("Single sign-on", "SAML, Enterprise only"), ("Exports", "PDF, CSV, scheduled email"),
               ("Query timeout", "120 seconds"), ("Cache lifetime", "15 minutes"),
               ("Audit log retention", "13 months")],
        planted=("TD-3309", "QUERY_ROW_LIMIT_EXCEEDED", "TIDE-CFG-query.max_rows",
                 "the result row limit", "a query returns more than 2,000,000 rows"),
    ),
}

HARDWARE_ISSUES = {
    "drive": ["WHEEL_SLIP", "MOTOR_OVERCURRENT", "ENCODER_MISMATCH", "BRAKE_RELEASE_FAIL", "MOTOR_OVERTEMP",
              "ODOMETRY_DRIFT", "CASTER_JAM", "DRIVE_CAN_TIMEOUT", "WHEEL_WEAR_LIMIT", "STALL_DETECTED"],
    "gripper": ["GRIPPER_VACUUM_LOW", "FINGER_POSITION_ERROR", "PAYLOAD_OVERWEIGHT", "CUP_LEAK_DETECTED",
                "WRIST_LIMIT_REACHED", "GRIP_FORCE_LOW", "PUMP_OVERTEMP", "VALVE_STUCK_OPEN",
                "ITEM_DROPPED", "TOOL_CHANGE_FAIL"],
    "lift deck": ["DECK_NOT_LEVEL", "LIFT_OVERLOAD", "LIFT_TIMEOUT", "DECK_POSITION_UNKNOWN",
                  "HYDRAULIC_PRESSURE_LOW", "LATCH_NOT_ENGAGED", "DECK_SENSOR_FAULT", "LOAD_SHIFTED",
                  "LIFT_MOTOR_OVERTEMP", "DECK_LIMIT_SWITCH"],
    "lift mast": ["MAST_POSITION_ERROR", "MAST_OVERLOAD", "BELT_TENSION_LOW", "MAST_LIMIT_REACHED",
                  "MAST_HOMING_FAIL", "CARRIAGE_JAM", "MAST_MOTOR_OVERTEMP", "MAST_SENSOR_FAULT",
                  "MAST_VIBRATION_HIGH", "MAST_BRAKE_FAULT"],
    "vision": ["CAMERA_NOT_DETECTED", "DEPTH_CALIBRATION_LOST", "LIDAR_WINDOW_DIRTY", "LOCALIZATION_LOST",
               "MARKER_NOT_FOUND", "IMAGE_TOO_DARK", "LIDAR_SPIN_FAULT", "POINT_CLOUD_SPARSE",
               "CAMERA_OVERTEMP", "TIME_SYNC_LOST"],
    "battery": ["BATTERY_LOW", "CELL_IMBALANCE", "CHARGE_CONTACT_FAULT", "PACK_OVERTEMP", "BMS_TIMEOUT",
                "CHARGE_RATE_LOW", "PACK_NOT_LATCHED", "CELL_UNDERVOLTAGE", "HEATER_FAULT", "PACK_ID_MISMATCH"],
    "safety scanner": ["SAFETY_FIELD_OBSTRUCTED", "SCANNER_CONTAMINATED", "SCANNER_CONFIG_MISMATCH",
                       "ESTOP_PRESSED", "SAFETY_RELAY_FAULT", "BUMPER_TRIGGERED", "SPEED_MONITOR_FAULT",
                       "SCANNER_TIMEOUT", "FIELD_SET_INVALID", "MUTING_TIMEOUT"],
    "network": ["WIFI_ROAM_FAIL", "FLEET_SERVER_UNREACHABLE", "CERTIFICATE_EXPIRED", "CLOCK_SKEW_HIGH",
                "MAP_SYNC_FAIL", "PACKET_LOSS_HIGH", "DNS_FAILURE", "FIRMWARE_MISMATCH",
                "TELEMETRY_BACKLOG", "VPN_TUNNEL_DOWN"],
}

SOFTWARE_ISSUES = {
    "ingestion": ["SOURCE_AUTH_FAILED", "SCHEMA_MISMATCH", "FILE_NOT_FOUND", "ROW_COUNT_DROP", "DUPLICATE_KEYS",
                  "LATE_ARRIVAL", "ENCODING_INVALID", "PARTITION_MISSING", "QUOTA_EXCEEDED", "CHECKSUM_FAILED"],
    "feature store": ["FEATURE_NULL_RATE_HIGH", "FEATURE_DRIFT", "BACKFILL_INCOMPLETE", "JOIN_EXPLOSION",
                      "FEATURE_VERSION_MISSING", "HOLIDAY_CALENDAR_STALE", "WEATHER_FEED_LATE",
                      "PROMO_CALENDAR_INVALID", "STORE_MAPPING_GAP", "TIMEZONE_MISMATCH"],
    "training": ["TRAINING_DIVERGED", "GPU_OUT_OF_MEMORY", "CHECKPOINT_CORRUPT", "BACKTEST_REGRESSION",
                 "TRAINING_TIMEOUT", "SPOT_INSTANCE_LOST", "SEED_NOT_SET", "DATA_LEAKAGE_DETECTED",
                 "HYPERPARAM_INVALID", "MODEL_CARD_MISSING"],
    "scoring": ["SCORING_FEATURES_STALE", "MODEL_ALIAS_MISSING", "PREDICTION_NEGATIVE", "SCORING_TIMEOUT",
                "OUTPUT_ROW_MISMATCH", "MODEL_VERSION_UNKNOWN", "SCORING_OOM", "CONFIDENCE_INTERVAL_INVALID",
                "STORE_NOT_SCORED", "SCORING_SKIPPED"],
    "export": ["SFTP_AUTH_FAILED", "EXPORT_FILE_TOO_LARGE", "API_RATE_LIMITED", "CSV_DELIMITER_CONFLICT",
               "EXPORT_LATE", "WEBHOOK_REJECTED", "ERP_MAPPING_MISSING", "EXPORT_TRUNCATED",
               "TOKEN_EXPIRED", "EXPORT_DUPLICATED"],
    "scheduler": ["JOB_MISSED", "JOB_OVERLAP", "DEPENDENCY_FAILED", "CRON_INVALID", "QUEUE_BACKLOG",
                  "WORKER_LOST", "RETRY_EXHAUSTED", "LOCK_HELD", "CALENDAR_CONFLICT", "PAUSED_BY_ADMIN"],
    "connector": ["WAREHOUSE_AUTH_FAILED", "CONNECTOR_TIMEOUT", "SCHEMA_DRIFT", "ROLE_MISSING_GRANT",
                  "NETWORK_POLICY_BLOCKED", "CONNECTOR_PAUSED", "CREDENTIAL_ROTATION_DUE", "TABLE_DROPPED",
                  "VIEW_INVALID", "REGION_MISMATCH"],
    "semantic layer": ["METRIC_DEFINITION_INVALID", "CIRCULAR_REFERENCE", "DIMENSION_MISSING",
                       "MEASURE_TYPE_MISMATCH", "FILTER_UNSUPPORTED", "JOIN_PATH_AMBIGUOUS",
                       "CALENDAR_NOT_SET", "CURRENCY_MISSING", "MODEL_NOT_PUBLISHED", "ALIAS_CONFLICT"],
    "query engine": ["QUERY_ROW_LIMIT_EXCEEDED", "QUERY_TIMEOUT", "QUERY_CANCELLED", "WAREHOUSE_SUSPENDED",
                     "SQL_COMPILE_ERROR", "RESULT_TOO_WIDE", "CONCURRENCY_LIMIT", "SPILL_TO_DISK",
                     "PERMISSION_DENIED", "QUERY_QUEUED_TOO_LONG"],
    "cache": ["CACHE_MISS_STORM", "CACHE_EVICTION_HIGH", "CACHE_STALE", "CACHE_KEY_COLLISION",
              "CACHE_WRITE_FAILED", "CACHE_DISABLED", "WARMUP_INCOMPLETE", "CACHE_SIZE_LIMIT",
              "CACHE_NODE_DOWN", "CACHE_VERSION_MISMATCH"],
    "sharing": ["SHARE_LINK_EXPIRED", "EMAIL_DIGEST_BOUNCED", "PDF_RENDER_FAILED", "EMBED_DOMAIN_BLOCKED",
                "SHARE_LIMIT_REACHED", "RECIPIENT_NOT_LICENSED", "SCHEDULE_INVALID", "ATTACHMENT_TOO_LARGE",
                "WATERMARK_FAILED", "EXPORT_DISABLED_BY_ADMIN"],
    "sso": ["SAML_ASSERTION_INVALID", "SSO_CERT_EXPIRED", "SCIM_SYNC_FAILED", "USER_NOT_PROVISIONED",
            "GROUP_MAPPING_MISSING", "IDP_UNREACHABLE", "SESSION_EXPIRED", "AUDIENCE_MISMATCH",
            "NAMEID_FORMAT_INVALID", "MFA_REQUIRED"],
}

CAUSES = [
    "a loose or corroded connector", "a worn component past its service interval",
    "a configuration change made outside the change process", "debris or dust on the sensor surface",
    "an expired credential", "a firmware or software version mismatch", "ambient temperature outside the rated range",
    "a recent site change that was not reflected in the configuration", "intermittent network loss",
    "a component damaged in transport or by impact",
]
FIRST_CHECKS = [
    "Check the event log for the five minutes before the error", "Confirm the unit is on the current release",
    "Inspect the connector and reseat it", "Clean the sensor surface with a dry microfibre cloth",
    "Compare the configuration with the last known-good snapshot", "Check the site network dashboard for drops",
    "Measure the supply voltage at the connector", "Run the built-in self-test from the service menu",
]


def manual(path: str, spec: dict, rng: random.Random) -> str:
    o = ORGS[spec["org"]]
    hardware = spec["org"] == "aurora"
    issues = HARDWARE_ISSUES if hardware else SOFTWARE_ISSUES
    kind = "service manual" if hardware else "operations guide"
    parts = [f"{spec['product']} — {kind}",
             f"Version {spec['version']}, {spec['year']}. Published by {o['company']}. "
             f"This edition replaces all earlier versions of the {spec['short']} {kind}.",
             "1. How to use this document",
             block("""
             This document is for the people who keep {short} running: {who}. Section 2 lists the
             key specifications. Section 3 covers safety and access. Section 4 is the reference of
             every error code, with what it means, the usual causes and what to do. Section 5 lists
             the {items} with their identifiers and service intervals. Section 6 has step-by-step
             procedures, and section 7 the maintenance schedule. When you open a ticket about
             {short}, always quote the error code and the identifier from these tables exactly as
             written: our tooling searches on them.
             """, short=spec["short"],
                   who=("field technicians and trained site staff" if hardware
                        else "support engineers, data engineers and customer administrators"),
                   items=("replaceable parts" if hardware else "configuration keys"))]
    parts.append("2. Key specifications")
    parts.append("\n".join(f"- {k}: {v}" for k, v in spec["specs"]))
    parts.append("3. Safety and access")
    parts.append(block("""
        {rules}
        Every intervention is recorded against a ticket in the {prefix} queue. A ticket id looks like
        {prefix}-{example}. Never close a ticket without recording what was changed, and never
        change two things at once: if the problem comes back, you need to know which change mattered.
        """, rules=("Lock out the unit with the physical key before opening any panel. Two people "
                    "are needed for any work above 1.5 m or any lift of more than 25 kg. Wear safety "
                    "shoes on the robot floor at all times." if hardware else
                    "Production changes need a second engineer to review them. Customer data stays "
                    "in the customer's own schema: never copy it to a laptop or a ticket. Use the "
                    "read-only role for investigation and the admin role only to apply a fix."),
             prefix=spec["ticket_prefix"], example=rng.randint(10000, 99999)))

    parts.append("4. Error code reference")
    code_p, name_p, part_p, what_p, cond_p = spec["planted"]
    planted_sub = code_p.split("-")[1][0]
    seen = set()
    for sub_no, sub in spec["subsystems"]:
        parts.append(f"4.{sub_no} {sub.capitalize()} errors")
        names = issues[sub]
        for i, name in enumerate(names):
            code = f"{spec['code_prefix']}-{sub_no}{rng.randint(0, 9)}{i:02d}"
            while code in seen:
                code = f"{spec['code_prefix']}-{sub_no}{rng.randint(0, 9)}{i:02d}"
            # Codes other documents quote (e.g. a postmortem) are fixed, so
            # the corpus never contradicts itself.
            code = spec.get("fixed_codes", {}).get(name, code)
            if sub_no == planted_sub and i == 0:
                code, name = code_p, name_p
                text = (f"{code} {name}. Raised when {cond_p}. Usual cause: wear or damage to "
                        f"{what_p}. Fix: replace or reset {what_p} ({part_p}) and run the self-test. "
                        f"If it returns within 24 hours, escalate to the {spec['short']} engineering team.")
            else:
                text = (f"{code} {name}. Raised by the {sub} subsystem. Usual cause: "
                        f"{rng.choice(CAUSES)}. First check: {rng.choice(FIRST_CHECKS).lower()}. "
                        f"Severity: {rng.choice(['warning', 'warning', 'error', 'critical'])}. "
                        f"{rng.choice(['Clears itself once the cause is fixed.', 'Needs a manual reset from the service menu.', 'Stops the unit until acknowledged.', 'Logged only; no action unless it repeats more than 3 times an hour.'])}")
            seen.add(code)
            parts.append(text)

    parts.append("4.7 Troubleshooting by symptom")
    for sub_no, sub in spec["subsystems"]:
        for symptom, advice in symptoms(sub, hardware, rng):
            parts.append(f"Symptom ({sub}): {symptom}\nWhat to do: {advice}")

    parts.append("5. " + ("Replaceable parts" if hardware else "Configuration keys"))
    for sub_no, sub in spec["subsystems"]:
        for j in range(18):
            if hardware:
                ident = f"{spec['part_prefix']}-{rng.randint(1000, 9999)}-{rng.choice('ABCD')}"
                if sub_no == planted_sub and j == 0:
                    ident = part_p
                parts.append(
                    f"{ident}: {sub} {rng.choice(['assembly', 'module', 'kit', 'sensor', 'cable set', 'bracket', 'seal kit'])}. "
                    f"Replace every {rng.choice([500, 1000, 2000, 4000, 8000])} operating hours or when worn. "
                    f"Fitting time about {rng.choice([10, 15, 20, 30, 45, 60])} minutes. "
                    f"Torque for fixings: {rng.choice([2.5, 4, 6, 9, 12])} Nm.")
            else:
                key = f"{sub.replace(' ', '_')}.{rng.choice(['max', 'min', 'default', 'retry', 'timeout'])}_{rng.choice(['rows', 'age_hours', 'attempts', 'seconds', 'workers', 'batch_size', 'percent'])}"
                ident = f"{spec['part_prefix']}-{key}"
                if sub_no == planted_sub and j == 0:
                    ident = part_p
                parts.append(
                    f"{ident}: controls the {sub} {rng.choice(['limit', 'timeout', 'retry budget', 'batch size', 'threshold'])}. "
                    f"Default {rng.choice([3, 5, 10, 30, 60, 120, 500, 2000])}. "
                    f"Changing it needs {rng.choice(['an admin role', 'a support engineer', 'a change ticket', 'customer approval'])}. "
                    f"Takes effect {rng.choice(['immediately', 'at the next nightly run', 'after a restart of the service', 'within 15 minutes'])}.")

    parts.append("6. Procedures")
    for n, (title, steps) in enumerate(procedures(spec, hardware, rng), start=1):
        parts.append(f"6.{n} {title}\n" + "\n".join(f"  {i}. {s}" for i, s in enumerate(steps, start=1)))

    parts.append("7. Maintenance schedule")
    for interval in (["Daily", "Weekly", "Every 500 operating hours", "Every 2,000 operating hours", "Yearly"]
                     if hardware else ["Every run", "Daily", "Weekly", "Monthly", "Quarterly"]):
        tasks = rng.sample([sub for _, sub in spec["subsystems"]], 3)
        parts.append(f"{interval}: check the {tasks[0]}; inspect the {tasks[1]}; review the {tasks[2]} "
                     f"event log for repeated warnings.")

    parts.append("8. Appendix: recent service history")
    parts.append(f"Resolved {spec['ticket_prefix']} tickets for {spec['short']}, quoted as examples "
                 f"of how common errors were fixed.")
    for _ in range(56):
        sub_no, sub = rng.choice(spec["subsystems"])
        issue = rng.choice(issues[sub])
        parts.append(
            f"{spec['ticket_prefix']}-{rng.randint(10000, 99999)} ({rng.choice(['Brightmart Columbus', 'Fernway Logistics Reno', 'Kettle & Co. Leeds', 'Ostrander Foods Dayton', 'Tidewater Parcel Norfolk', 'Northgate Grocers', 'Halcyon Retail', 'Pine & Petal', 'Marlow Outfitters', 'Sundial Pharmacy'])}): "
            f"{issue.replace('_', ' ').lower()} on the {sub}. Cause found: {rng.choice(CAUSES)}. "
            f"Resolved in {rng.randint(1, 9)} hours {rng.choice(['by a remote fix', 'on site', 'after a configuration rollback', 'by replacing the affected part', 'after a release'])}.")

    fact(path, "exact", f"{code_p} {name_p}: {cond_p}; fix with {part_p}",
         f"What does {code_p} mean on {spec['short']} and which part fixes it?")
    fact(path, "versioned", f"{spec['short']} {kind} version {spec['version']} ({spec['year']})",
         f"Which version of the {spec['short']} {kind} is current?")
    return "\n\n".join(parts)


SYMPTOMS_HW = [
    ("The robot keeps letting go of soft plastic bags just after lifting them.",
     "Check the {sub} for leaks around the cups and for a pump that sounds weaker than usual; see the error reference for the matching code."),
    ("The robot moves in a slightly curved line when it should go straight.",
     "Look at the {sub} first: uneven wear or a dirty sensor is the usual reason. Recalibrate after cleaning."),
    ("It stops for no obvious reason in the same aisle every day.",
     "Compare the stop positions in the event log with the floor map; something on the floor or a reflective surface may be confusing the {sub}."),
    ("It takes much longer than usual to be ready after charging.",
     "Check the {sub} temperature readings; a pack that is too cold or too hot charges slowly by design."),
    ("The status light blinks amber but the tablet shows no error.",
     "Open the service menu and read the {sub} warnings, which do not always raise a visible error."),
    ("Staff report a grinding or clicking sound during moves.",
     "Stop the robot, lock it out and inspect the {sub} for debris and worn bearings."),
]
SYMPTOMS_SW = [
    ("Customers say their numbers look a day behind.",
     "Check the {sub} run history for the last two nights; a silent failure usually shows as a missing run, not an error."),
    ("A dashboard that used to be quick has become slow over a few weeks.",
     "Look at the {sub} for growth in data volume or a filter that stopped using its cache."),
    ("One store's figures jump wildly from day to day.",
     "Check the {sub} input for that store for duplicated or missing days before suspecting the model."),
    ("A customer can sign in but sees an empty workspace.",
     "Check their group mappings and permissions in the {sub} settings."),
    ("Exports arrive, but some rows are missing at the end of the file.",
     "Compare row counts between the {sub} output and the export; a size limit may be cutting the file."),
    ("Nightly jobs finish later every week.",
     "Review the {sub} timings and queue depth; a growing backlog points to too few workers."),
]


def symptoms(sub: str, hardware: bool, rng: random.Random) -> list[tuple[str, str]]:
    pool = SYMPTOMS_HW if hardware else SYMPTOMS_SW
    return [(s, a.format(sub=sub)) for s, a in rng.sample(pool, 4)]


def procedures(spec: dict, hardware: bool, rng: random.Random) -> list[tuple[str, list[str]]]:
    s = spec["short"]
    if hardware:
        base = [
            ("Replacing the battery pack", [
                f"Drive {s} to the service bay and stop it with the service tablet.",
                "Turn the key to LOCK and wait for the status light to turn solid blue.",
                "Release the two pack latches on the left side.",
                "Slide the pack out on the rails; it weighs about 24 kg, so use the lift trolley.",
                "Slide the new pack in until both latches click.",
                "Turn the key to RUN and check the pack serial number in Skyview.",
                "Return the old pack to the take-back crate and log its serial on the ticket."]),
            ("Calibrating the depth cameras", [
                "Place the checkerboard target 1.0 m in front of the robot, centred.",
                "Open Service > Vision > Calibrate on the tablet.",
                "Hold the target still while the progress bar fills, about 90 seconds.",
                "Move the target to 1.5 m and repeat.",
                "Accept the result only if the reprojection error shown is under 0.4 pixels."]),
            ("Recovering from a lost localization", [
                "Stop the robot and check that the lidar window is clean.",
                "Drive it by hand to the nearest charging dock, which is a known position.",
                "Choose Service > Navigation > Relocalize.",
                "If relocalization fails twice, re-run floor-map alignment and raise a ticket."]),
            ("Updating firmware on site", [
                "Check the release notes for the target version and any special steps.",
                "Dock the robot and keep it charging during the update.",
                "Start the update from Skyview, never from a USB stick.",
                "Do not interrupt power for the 12 minutes the update takes.",
                "Run the self-test and send one test mission before returning the robot to service."]),
        ]
    else:
        base = [
            ("Re-running a failed nightly job", [
                "Find the failed run in the scheduler view and note its run id.",
                "Read the first error in the log, not the last; later errors are usually consequences.",
                "Fix the cause (credentials, schema, quota) before re-running.",
                "Re-run with the same run date so outputs land in the right partition.",
                "Tell the customer if their data was more than 6 hours late."]),
            ("Rotating a customer's warehouse credentials", [
                "Ask the customer administrator to create the new key pair in their warehouse.",
                "Paste the public key into the connector settings; never ask for the private key by email.",
                "Run the connection test.",
                "Remove the old key only after a successful nightly run with the new one."]),
            ("Rolling back a configuration change", [
                "Open the configuration history and find the last known-good version.",
                "Compare it with the current version and confirm the change you want to undo.",
                "Restore it and record the ticket id in the change note.",
                "Watch the next scheduled run end to end."]),
            ("Onboarding a new customer tenant", [
                "Create the tenant schema from the template; never copy another customer's schema.",
                "Set the time zone, currency and fiscal calendar before loading any data.",
                "Load two years of history and run the data-quality report.",
                "Agree the first go-live date with the customer only after the report is clean."]),
        ]
    extra = []
    for sub_no, sub in spec["subsystems"]:
        extra.append((f"Investigating repeated {sub} errors", [
            f"List every {sub} error from the last 7 days and group them by code.",
            rng.choice(FIRST_CHECKS) + ".",
            f"Check whether the errors started after a release or a {rng.choice(['site', 'configuration', 'network', 'schema'])} change.",
            "Fix one cause at a time and watch for 24 hours before closing the ticket."]))
    return base + extra


# =============================================================================
# Contracts (one per org, from a shared clause library with different terms)
# =============================================================================

CONTRACTS = {
    "aurora/cellcore-supply-agreement.txt": dict(
        org="aurora", title="Battery Supply Agreement", number="AUR-LGL-2025-114", year=2026,
        party_a="Aurora Robotics, Inc.", party_b="CellCore Energy Ltd.", role_a="Buyer", role_b="Supplier",
        start="1 January 2026", end="31 December 2028", subject="Atlas battery packs (rev. D)",
        price="2,150 US dollars per pack for the first 1,000 packs in each contract year, and 1,980 US "
              "dollars per pack above that volume",
        minimum="600 packs per contract year", penalty="12 percent of the value of the unmet volume",
        warranty="3 years or 3,000 charge cycles, whichever comes first",
        notice="9 months' written notice, given no earlier than 1 January 2027",
        payment_days=45, liability_cap="the fees paid in the 12 months before the claim",
        law="the Commonwealth of Pennsylvania", service_level="98 percent of orders delivered on time",
        credit="2 percent of the order value for each full week of delay, up to 10 percent"),
    "bluefin/northgate-master-services-agreement.txt": dict(
        org="bluefin", title="Master Services Agreement", number="BLU-MSA-2024-031", year=2026,
        party_a="Bluefin Analytics, Inc.", party_b="Northgate Grocers LLC", role_a="Provider",
        role_b="Customer", start="1 April 2024", end="31 March 2027",
        subject="the Tide dashboards (Enterprise) and Current forecasting services",
        price="4,500 US dollars per month for Tide Enterprise, and 18 US dollars per store per month "
              "for Current, with a 15 percent discount on stores above 1,000",
        minimum="800 stores", penalty="the monthly fee for any store below the minimum",
        warranty="that the services will perform materially as described in the documentation",
        notice="90 days' written notice before the end of the term",
        payment_days=30, liability_cap="the fees paid in the 12 months before the claim",
        law="the Commonwealth of Massachusetts", service_level="99.9 percent monthly availability",
        credit="5 percent of the monthly fee for each 0.1 percentage point below the service level, "
               "up to 25 percent; forecasts delivered more than 12 hours late count as unavailable"),
}

CLAUSES = [
    ("Definitions", """
     In this agreement, "Agreement" means this document and its schedules; "Effective Date" means
     {start}; "Contract Year" means each twelve-month period starting on the Effective Date or an
     anniversary of it; "Confidential Information" means any information disclosed by one party to
     the other that is marked confidential or that a reasonable person would understand to be
     confidential; and "Business Day" means a day other than a Saturday, Sunday or public holiday in
     the place of the party receiving a notice.
     """),
    ("Scope", """
     The {role_b_or_a} shall provide, and the {role_a_or_b} shall buy, {subject} on the terms of this
     Agreement. Each order or statement of work forms part of this Agreement once both parties have
     signed it. If an order conflicts with this Agreement, this Agreement prevails unless the order
     expressly says which clause it changes.
     """),
    ("Term", """
     This Agreement starts on {start} and ends on {end}, unless ended earlier under the termination
     clause. Either party may end this Agreement for convenience by giving {notice}.
     """),
    ("Prices", """
     The prices are {price}. Prices are exclusive of sales tax and similar duties, which are added at
     the applicable rate. Prices may not be increased during a Contract Year. Any increase for a
     following Contract Year must be notified at least 60 days before that Contract Year starts and
     may not exceed 6 percent.
     """),
    ("Minimum commitment", """
     The minimum commitment is {minimum}. If the commitment is not met in a Contract Year, the
     shortfall is charged at {penalty}, invoiced within 30 days after the end of that Contract Year.
     """),
    ("Invoicing and payment", """
     Invoices are issued monthly in arrears and are payable within {payment_days} days of receipt.
     Undisputed amounts that are paid late bear interest at 1 percent per month. A party that
     disputes an invoice in good faith must say so in writing within 15 days, and the undisputed part
     remains payable on time.
     """),
    ("Service levels and credits", """
     The service level is {service_level}. If it is not met in a month, the affected party is
     entitled to a credit of {credit}. Credits are the sole financial remedy for a missed service
     level, but do not limit the right to terminate for repeated failure under the termination clause.
     Credits must be claimed within 60 days after the end of the month concerned.
     """),
    ("Warranty", """
     The {role_b_or_a} warrants {warranty}. If a warranty claim is upheld, the {role_b_or_a} shall at
     its option repair, replace or re-perform, or refund the price paid for the affected items.
     The warranty does not cover misuse, accidents, or modification by anyone other than the
     {role_b_or_a}.
     """),
    ("Confidentiality", """
     Each party shall keep the other's Confidential Information confidential, use it only to perform
     this Agreement, and share it only with employees and advisers who need to know it and are bound
     by similar duties. These duties last during this Agreement and for five years after it ends,
     and for trade secrets for as long as they remain trade secrets.
     """),
    ("Data protection", """
     Where a party processes personal data on behalf of the other, it shall do so only on documented
     instructions, keep it secure, notify the other party of any personal data breach without undue
     delay and in any case within 48 hours of becoming aware of it, and delete or return it at the
     end of this Agreement.
     """),
    ("Intellectual property", """
     Each party keeps the intellectual property it owned before this Agreement. Nothing in this
     Agreement transfers ownership of either party's intellectual property, except as expressly
     stated in an order. Feedback may be used freely by the party receiving it.
     """),
    ("Liability", """
     Neither party excludes liability for death or personal injury caused by negligence, for fraud,
     or for anything that cannot be excluded by law. Otherwise, each party's total liability under
     this Agreement in any Contract Year is limited to {liability_cap}. Neither party is liable for
     loss of profit, loss of data, or indirect or consequential loss.
     """),
    ("Insurance", """
     Each party shall maintain, with reputable insurers, general liability insurance of at least
     5 million US dollars per claim, and product or professional liability insurance appropriate to
     its obligations, and shall provide certificates on request.
     """),
    ("Termination", """
     Either party may terminate this Agreement with immediate effect by written notice if the other
     party commits a material breach that is not remedied within 30 days of a notice describing it,
     or if the other party becomes insolvent. A failure to meet the service level in three
     consecutive months is a material breach.
     """),
    ("Consequences of termination", """
     On termination, all unpaid invoices become payable, each party returns or destroys the other's
     Confidential Information, and the clauses on confidentiality, liability, and governing law
     continue to apply.
     """),
    ("Force majeure", """
     Neither party is liable for a delay or failure caused by events beyond its reasonable control,
     such as natural disasters, war, or the failure of public utilities, provided it notifies the
     other party promptly and uses reasonable efforts to limit the effect. If such an event lasts more
     than 60 days, either party may terminate the affected order.
     """),
    ("Assignment and subcontracting", """
     Neither party may assign this Agreement without the other's written consent, which may not be
     unreasonably withheld, except to a successor of all or substantially all of its business. A
     party remains responsible for the acts of its subcontractors.
     """),
    ("Notices", """
     Notices must be in writing and sent by email to the contacts named in the schedule, with a copy
     by courier for notices of breach or termination. A notice sent by email is received on the next
     Business Day after it is sent.
     """),
    ("Governing law and disputes", """
     This Agreement is governed by the laws of {law}. The parties shall first try to resolve any
     dispute through their executive sponsors within 30 days; failing that, the courts of {law} have
     exclusive jurisdiction.
     """),
    ("Orders and forecasts", """
     The {role_a_or_b} shall give a rolling forecast of its expected needs for the next six months,
     updated by the fifth Business Day of each month. Forecasts are for planning only and are not
     orders, except that the first month of each forecast is binding. Orders must be confirmed or
     rejected within three Business Days; an order not rejected in that time is accepted.
     """),
    ("Delivery and acceptance", """
     Delivery is made to the address in the order. The {role_a_or_b} shall inspect deliveries within
     ten Business Days and notify any visible defect or shortfall in writing in that time; after that,
     the delivery is treated as accepted, except for defects that a reasonable inspection would not
     have revealed. Rejected items are collected and replaced at the {role_b_or_a}'s cost.
     """),
    ("Quality and audits", """
     The {role_b_or_a} shall maintain a quality management system certified to a recognised standard
     and shall keep records of production, testing and incidents for at least six years. The
     {role_a_or_b} may audit those records once a year, on twenty Business Days' notice, during
     normal working hours, and at its own cost unless the audit finds a material non-compliance.
     """),
    ("Change control", """
     Either party may propose a change to the scope, the specification or the service levels by a
     written change request. The other party shall respond within fifteen Business Days with its
     assessment of the effect on price, timing and risk. No change is binding until a change note is
     signed by both parties' authorised representatives.
     """),
    ("Security", """
     The {role_b_or_a} shall keep the systems it uses to perform this Agreement secure, in line with
     good industry practice, including access control, encryption of data in transit and at rest,
     vulnerability management and staff training, and shall notify the {role_a_or_b} of any security
     incident affecting it within 48 hours.
     """),
    ("Business continuity", """
     The {role_b_or_a} shall maintain and test, at least once a year, a business continuity plan
     that allows it to resume the services or deliveries within five Business Days of a serious
     disruption, and shall share a summary of each test with the {role_a_or_b}.
     """),
    ("Personnel", """
     Each party is responsible for its own staff, including their pay, taxes and benefits. Staff
     working at the other party's premises shall follow its site rules and safety instructions.
     Neither party shall, during this Agreement and for twelve months after it, actively solicit for
     employment any employee of the other party who worked on this Agreement.
     """),
    ("Compliance with laws", """
     Each party shall comply with all laws that apply to it in performing this Agreement, including
     laws on anti-bribery, sanctions, export controls and modern slavery. Each party shall tell the
     other promptly if it becomes aware of a breach of this clause connected with this Agreement.
     """),
    ("Publicity", """
     Neither party shall issue a press release or make any public statement about this Agreement
     without the other party's prior written approval, except that each may name the other as a
     customer or supplier in a list of customers or suppliers.
     """),
    ("Transition at the end of the Agreement", """
     For up to ninety days after this Agreement ends, the {role_b_or_a} shall provide reasonable
     help to move the services or supply to the {role_a_or_b} or to a new provider, at the rates in
     force at the end of this Agreement, and shall return all data and materials belonging to the
     {role_a_or_b}.
     """),
    ("Escalation", """
     Operational issues are raised first between the account managers named in the schedule. An
     issue not resolved within ten Business Days is escalated to each party's head of operations,
     and then, after a further ten Business Days, to the executive sponsors.
     """),
    ("Entire agreement", """
     This Agreement is the entire agreement between the parties about its subject and replaces any
     earlier agreement or understanding. Changes are valid only if made in writing and signed by
     both parties. If any clause is found invalid, the rest of this Agreement continues in effect.
     """),
]


def contract(path: str, c: dict) -> str:
    p = dict(c)
    # In the supply agreement the supplier performs; in the MSA the provider does.
    p["role_b_or_a"] = c["role_b"] if c["role_b"] == "Supplier" else c["role_a"]
    p["role_a_or_b"] = c["role_a"] if c["role_b"] == "Supplier" else c["role_b"]
    parts = [f"{c['title']} — contract number {c['number']}",
             f"Version: as amended, consolidated text for {c['year']}. Between {c['party_a']} "
             f"(the {c['role_a']}) and {c['party_b']} (the {c['role_b']}).",
             "Recitals",
             block("""
             The parties wish to set out the terms on which {subject} will be supplied, including
             prices, commitments, service levels and the allocation of risk. This consolidated text
             includes every amendment signed up to the date shown above.
             """, **p)]
    for n, (title, body) in enumerate(CLAUSES, start=1):
        parts.append(f"Clause {n}. {title}")
        text = block(body, **p)
        # Sub-clauses make the text read like the real thing and give exact
        # clause references (e.g. "clause 7.2") to search for.
        sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
        parts.append("\n".join(f"{n}.{i} {s.rstrip('.')}." for i, s in enumerate(sentences, start=1)))
    parts.append("Schedule 1. Contacts for notices")
    parts.append(f"{c['party_a']}: legal@{ORGS[c['org']]['domain']}. {c['party_b']}: contracts desk, "
                 f"as named in the latest signed order.")
    parts.append("Schedule 2. Service-level reporting")
    parts.append(block("""
        The {role_b_or_a} shall send a monthly report by the tenth Business Day of each month showing,
        for the month before: performance against the service level ({service_level}), every incident
        with its start, end and cause, credits due ({credit}), open change requests, and planned
        changes for the next month. Reports are reviewed in a monthly service meeting; the minutes are
        kept by the {role_a_or_b}.
        """, **p))
    parts.append("Schedule 3. Amendments")
    parts.append(f"Amendment 1 (signed {c['start']}): original terms. Amendment 2: updated the "
                 f"service-level credits to {c['credit']}.")
    fact(path, "exact", f"Contract {c['number']}: payment within {c['payment_days']} days; service level {c['service_level']}",
         f"What payment terms does contract {c['number']} set?")
    fact(path, "paraphrase", f"Termination for convenience: {c['notice']}",
         f"How much warning does either side need to give to walk away from the {c['party_b']} deal?")
    return "\n\n".join(parts)


# =============================================================================
# Long postmortems (consistent with the short summaries already in kb/docs/)
# =============================================================================

def postmortem_aurora(rng: random.Random) -> str:
    name = "aurora/postmortem-2026-08-atlas-fleet-stall-full-report.txt"
    lines = [
        "Incident report INC-2026-0819-01 — Atlas fleet stall at Brightmart Columbus",
        "Version 1.3 (final), 2026. Severity SEV1. Duration 2 hours 14 minutes (06:12 to 08:26 Eastern, "
        "19 August 2026). Author: Mei Tanaka, platform team. Reviewed by Dana Okafor.",
        "This is the full report. A one-page summary is kept separately as "
        "postmortem-2026-08-atlas-fleet-stall.",
        "1. Summary",
        block("""
        At 06:12 Eastern on 19 August 2026, 63 of the 140 Atlas movers at the Brightmart distribution
        center in Columbus, Ohio stopped and refused new missions. Picking throughput at the site fell
        by about 45 percent. The robots stopped safely: no person or product was harmed. Service was
        fully restored at 08:26. The trigger was a floor-map update pushed at 06:05 that contained a
        no-go zone drawn with an inverted polygon, which our map validator accepted because it only
        checked that polygons were closed. Recovery took longer than it should have because the
        previous map had already been removed from the site cache, so it had to be rebuilt and pushed
        again (about 70 minutes of the outage).
        """),
        "2. Impact",
        block("""
        - 63 of 140 Atlas movers stopped; the other 77 kept running because they were outside the
          inverted zone when the map arrived or had not yet downloaded it.
        - Throughput at the site: about 45 percent below plan between 06:15 and 08:30.
        - Customer orders delayed: Brightmart estimated 3,100 outbound orders shipped late.
        - Commercial: we offered Brightmart a service credit of 15,000 US dollars (ticket FOPS-88213).
        - No safety incidents. All stops were controlled stops, logged as A-5205 MISSION_REJECTED_NOGO.
        """),
        "3. Timeline (all times Eastern, 19 August 2026)",
    ]
    t = [(6, 5, "Map version 2026-08-19.02 for the Columbus site is published by the site-mapping tool."),
         (6, 9, "Robots begin downloading the new map as they finish their current missions."),
         (6, 12, "First robots reject new missions with A-5205 MISSION_REJECTED_NOGO."),
         (6, 14, "Alert atlas-missions-rejected fires; page sent to the platform primary on call."),
         (6, 17, "Primary on call, Sam Whitfield, acknowledges the page (3 minutes after it fired)."),
         (6, 24, "Field-ops lead on site confirms robots are stopped safely and the floor is clear."),
         (6, 31, "Incident declared SEV1; status page updated; Brightmart account manager informed."),
         (6, 38, "Map change identified as the likely trigger by comparing the mission rejections with the map publish time."),
         (6, 44, "Attempted rollback to map 2026-08-19.01 fails: the version is not in the site cache."),
         (6, 52, "Decision to rebuild 2026-08-19.01 from the mapping tool's history."),
         (7, 30, "Rebuilt map validated manually by two engineers."),
         (7, 41, "Rebuilt map pushed to the site."),
         (7, 58, "First stopped robots accept missions again."),
         (8, 26, "All 140 robots running; throughput back to plan by 08:40. Incident resolved."),
         (9, 15, "Brightmart operations director briefed by phone.")]
    for h, m, what in t:
        lines.append(f"{h:02d}:{m:02d} — {what}")
    lines.append("4. Detailed timeline from robot telemetry")
    robots = [f"ATL-{rng.randint(10000, 99999)}" for _ in range(63)]
    # Each stopped robot, then the 30 that stopped a second time while the
    # rebuilt map was rolling out.
    for i, r in enumerate(robots + robots[:30]):
        minute = 12 + ((i % 63) * 46) // 63
        lines.append(f"06:{minute:02d} robot {r} rejected mission M-{rng.randint(400000, 499999)} with "
                     f"A-5205 MISSION_REJECTED_NOGO; position inside zone NZ-{rng.randint(10, 99)}; "
                     f"recovered at {7 + (58 + i) // 60:02d}:{(58 + i) % 60:02d}.")
    lines += [
        "5. Root cause",
        block("""
        The site-mapping tool lets a mapper draw no-go zones as polygons. A polygon drawn clockwise is
        treated as an exclusion around its inside; drawn counter-clockwise, the robots' planner treats
        it as an exclusion of everything outside it. A mapper redrawing the zone around a new charging
        area drew it counter-clockwise. Our map validator checked that every polygon was closed and
        inside the site boundary, but not its orientation, so the map was published.

        Contributing factor: the site cache keeps only two map versions, and the cache had pruned
        2026-08-19.01 after two quick publishes earlier that morning (2026-08-18.03 and 2026-08-19.01
        were both test publishes). Rolling back therefore meant rebuilding the map, not switching to it.
        """),
        "6. What went well",
        "- The page was acknowledged within 3 minutes.\n- Robots stopped safely, as designed.\n"
        "- The field-ops lead on site coordinated manual picking for the most urgent orders.",
        "7. What went badly",
        "- Recovery took 70 minutes longer than a rollback should.\n- The validator had a known gap "
        "(ticket PLAT-4412, opened in March 2026) that had not been prioritised.\n- The status page was "
        "updated 19 minutes after the page, against a target of 15 minutes.",
        "8. Action items",
    ]
    actions = [("Validate polygon orientation in the map validator", "Mei Tanaka", "PLAT-4412", "15 September 2026"),
               ("Keep the last 5 map versions in every site cache", "Luis Ferreira", "PLAT-4520", "30 September 2026"),
               ("Offer Brightmart a 15,000 USD service credit", "account team", "FOPS-88213", "26 August 2026"),
               ("Add a dry-run of every map publish against the fleet planner", "Mei Tanaka", "PLAT-4521", "31 October 2026"),
               ("Status page within 15 minutes: add it to the SEV1 checklist", "Dana Okafor", "PLAT-4522", "5 September 2026")]
    for what, who, ticket, due in actions:
        lines.append(f"- {what}. Owner: {who}. Ticket {ticket}. Due {due}.")
    lines += extra_postmortem_sections("aurora")
    lines.append("13. Appendix: log excerpt")
    for i in range(120):
        lines.append(f"2026-08-19T06:{12 + i // 3:02d}:{rng.randint(0, 59):02d}-04:00 planner "
                     f"WARN A-5205 MISSION_REJECTED_NOGO robot={rng.choice(robots)} zone=NZ-{rng.randint(10, 99)} "
                     f"map=2026-08-19.02")
    fact(name, "exact", "Brightmart service credit ticket FOPS-88213; validator gap ticket PLAT-4412",
         "Which ticket tracked the service credit for the Atlas fleet stall?")
    fact(name, "paraphrase", "The no-go polygon was drawn counter-clockwise, so the planner excluded everything outside it",
         "Why did the robots think almost the whole warehouse was off limits?")
    return name, "\n\n".join(lines)


def postmortem_bluefin(rng: random.Random) -> str:
    name = "bluefin/postmortem-2026-07-forecast-outage-full-report.txt"
    lines = [
        "Incident report BLU-INC-2026-0722 — stale forecasts for 11 customers",
        "Version 2.0 (final), 2026. Severity P1. Customer impact: 19 hours of stale forecasts "
        "(02:00 to 21:00 Eastern, 22 July 2026). Author: Priya Raman, ML team. Reviewed by the data-eng lead.",
        "This is the full report; the short summary is postmortem-2026-07-forecast-outage.",
        "1. Summary",
        block("""
        The nightly forecast run at 02:00 Eastern on 22 July 2026 failed for 11 customers, including
        Northgate Grocers, our largest account. Their Tide dashboards kept showing the previous day's
        forecasts until fresh ones were published at 21:00. Nobody at Bluefin was paged: the failure
        alert went to a Slack channel that had been archived in June, and the first we heard was from
        Northgate's planning team at 07:40. The trigger was a schema change to the promotions table,
        which renamed the column promo_type to promotion_type; the forecasting job still read the old
        name and failed with CUR-2215 PROMO_CALENDAR_INVALID.
        """),
        "2. Impact",
        block("""
        - 11 customers received stale forecasts for one day; 4 of them reordered from stale numbers.
        - Northgate Grocers estimated 214 stores ordered from the previous day's forecast.
        - We credited Northgate 8,000 US dollars (ticket SUP-55102) under clause 7 of the master
          services agreement, which treats forecasts more than 12 hours late as unavailable.
        """),
        "3. Timeline (Eastern time, 22 July 2026)",
    ]
    t = [(1, 12, "Pull request #2291 to the dbt project, renaming promo_type to promotion_type, is merged."),
         (1, 40, "Nightly dbt build applies the rename to the promotions table for all tenants."),
         (2, 0, "Nightly forecast run starts."),
         (2, 6, "Run fails for 11 tenants with CUR-2215 PROMO_CALENDAR_INVALID; alert posted to #ml-alerts-old (archived)."),
         (7, 40, "Northgate's planning team reports stale forecasts through the Enterprise support line."),
         (7, 52, "Support on duty opens SUP-55102 and escalates to the ML on-call as P1."),
         (8, 15, "ML on-call identifies the column rename in the failing query."),
         (8, 40, "Decision: patch the forecasting job to read the new name rather than revert the rename."),
         (11, 5, "Patched job released after review."),
         (11, 20, "Re-run started for the 11 tenants with run date 22 July."),
         (20, 10, "Re-run completes; quality checks pass."),
         (21, 0, "Forecasts published to Tide and exports for all 11 tenants. Incident resolved."),
         (21, 30, "Customer success sends incident notices to the 11 customers.")]
    for h, m, what in t:
        lines.append(f"{h:02d}:{m:02d} — {what}")
    lines.append("4. Affected tenants and runs")
    tenants = ["Northgate Grocers"] + [f"tenant-{rng.randint(100, 999)}" for _ in range(10)]
    for tn in tenants:
        lines.append(f"{tn}: run {rng.randint(700000, 799999)} failed at 02:0{rng.randint(1, 9)} with "
                     f"CUR-2215 PROMO_CALENDAR_INVALID; re-run {rng.randint(800000, 899999)} succeeded at "
                     f"20:{rng.randint(10, 59):02d}; stores affected {rng.randint(40, 900)}.")
    lines += [
        "5. Root cause",
        block("""
        The promotions table had no schema contract. The data-eng team renamed a column to match a new
        naming standard and checked every dbt model that used it, but the forecasting job reads the
        table directly, outside dbt, so nothing flagged it. Separately, the forecast-failure alert was
        configured to post to a Slack channel that was archived during a channel clean-up in June; the
        alert succeeded technically, so nothing reported that it had nowhere to go.
        """),
        "6. Action items",
    ]
    actions = [("Schema contracts on every table the forecasting job reads", "data-eng", "DE-1873", "30 September 2026"),
               ("Route the forecast-failure alert to PagerDuty, not Slack", "ML lead", "ML-0942", "29 July 2026"),
               ("Credit 8,000 USD to Northgate Grocers", "finance", "SUP-55102", "5 August 2026"),
               ("Alert when an alert's destination channel is archived", "platform", "PLT-0311", "31 August 2026")]
    for what, who, ticket, due in actions:
        lines.append(f"- {what}. Owner: {who}. Ticket {ticket}. Due {due}.")
    lines += extra_postmortem_sections("bluefin")
    lines.append("11. Appendix: scheduler log excerpt")
    for i in range(120):
        lines.append(f"2026-07-22T02:0{rng.randint(0, 9)}:{rng.randint(0, 59):02d}-04:00 scheduler ERROR "
                     f"CUR-2215 PROMO_CALENDAR_INVALID tenant={rng.choice(tenants[1:])} "
                     f"column=promo_type not found in promotions")
    fact(name, "exact", "Northgate credit ticket SUP-55102; failing code CUR-2215 PROMO_CALENDAR_INVALID",
         "Which error code did the forecast run fail with on 22 July 2026?")
    fact(name, "paraphrase", "The failure alert went to a Slack channel that had been archived",
         "Why did nobody at Bluefin find out the forecasts had failed until the customer called?")
    return name, "\n\n".join(lines)


POSTMORTEM_EXTRA = {
    "aurora": dict(site="Brightmart Columbus", detect="3 minutes", status="19 minutes", n0=9),
    "bluefin": dict(site="Northgate Grocers and 10 other tenants", detect="5 hours 40 minutes",
                    status="2 hours 25 minutes", n0=7),
}


def extra_postmortem_sections(org: str) -> list[str]:
    p = POSTMORTEM_EXTRA[org]
    n = p["n0"]
    return [
        f"{n}. Detection",
        block("""
        Time from the first failure to a human knowing about it: {detect}. We measure detection from the
        first failed operation, not from the first alert, because an alert that nobody sees does not
        count. Our target for this class of incident is 10 minutes. The gap is discussed in the root
        cause and the action items.
        """, **p),
        f"{n + 1}. Response",
        block("""
        The incident commander role was taken by the first engineer who acknowledged the incident and
        handed to the team lead once they joined. Two channels were used: one for the technical
        response and one for customer communication, which kept the technical channel readable. The
        first status-page update was posted {status} after detection. Engineers rotated every two hours
        to avoid tired decisions, and every decision with customer impact was written down in the
        incident document with its time and who made it.
        """, **p),
        f"{n + 2}. Five whys",
        block("""
        Why did customers see an outage? Because the change reached production unchecked.
        Why was it unchecked? Because the validation step did not cover this kind of change.
        Why did it not cover it? Because nobody had written down which changes it was meant to catch.
        Why was that not written down? Because the step was added in a hurry after an earlier incident
        and never revisited.
        Why was it never revisited? Because we had no regular review of our safety checks. The new
        quarterly review of safety checks closes that gap.
        """, **p),
        f"{n + 3}. Customer communication",
        block("""
        Customers affected ({site}) received a first notice during the incident, an update when service
        was restored, and this report's summary within five business days. Every notice said what
        happened, what it meant for them, and what we were doing, in plain language and without blaming
        anyone. Account managers followed up by phone with every affected customer.
        """, **p),
    ]


# =============================================================================
# Release notes (versioned items with exact ids)
# =============================================================================

def release_notes(org: str, rng: random.Random) -> tuple[str, str]:
    if org == "aurora":
        name, title = "aurora/firmware-release-notes-2026.txt", "Aurora firmware release notes — 2026"
        versions = [f"2026.{n}.0" for n in range(1, 12)]
        prefix, products = "FW", ["Pika", "Atlas"]
        areas = list(HARDWARE_ISSUES)
    else:
        name, title = "bluefin/release-notes-2026.txt", "Bluefin release notes — 2026"
        versions = [f"Tide 7.{n} / Current 4.{min(n // 2, 6)}" for n in range(1, 12)]
        prefix, products = "BLU", ["Tide", "Current"]
        areas = list(SOFTWARE_ISSUES)
    parts = [title, f"Version {len(versions)} of this document, 2026: one entry per release from January "
                    f"to November 2026. Each item carries its tracking id."]
    verbs = ["Fixed", "Improved", "Added", "Changed", "Removed"]
    for i, v in enumerate(versions):
        month = ["January", "February", "March", "April", "May", "June", "July", "August",
                 "September", "October", "November"][i]
        parts.append(f"Release {v} — {rng.randint(3, 27)} {month} 2026")
        items = []
        for _ in range(rng.randint(18, 24)):
            area = rng.choice(areas)
            issue = rng.choice((HARDWARE_ISSUES if org == "aurora" else SOFTWARE_ISSUES)[area])
            items.append(f"- {prefix}-{rng.randint(1000, 9999)} {rng.choice(verbs)}: {rng.choice(products)} "
                         f"{area} — {issue.replace('_', ' ').lower()} "
                         f"{rng.choice(['is handled more gracefully', 'now has a clearer message', 'no longer needs a manual reset', 'is reported to the fleet dashboard', 'threshold adjusted', 'retries with backoff', 'is logged with more context'])}.")
        parts.append("\n".join(items))
        parts.append(f"Known issues in {v}: {rng.choice(['none', 'occasional slow start after an update', 'a cosmetic error in the status page', 'one rare timeout under heavy load, fixed in the next release'])}.")
    planted = ("FW-7731", "2026.9.0", "the Pika gripper no longer drops polybags when vacuum recovers late") \
        if org == "aurora" else ("BLU-6620", "Tide 7.9", "dashboards no longer double-count returns in net sales")
    parts.append(f"Highlight of the year: {planted[0]} in {planted[1]}: {planted[2]}.")
    fact(name, "exact", f"{planted[0]} shipped in {planted[1]}: {planted[2]}",
         f"Which release fixed {planted[0]}?")
    return name, "\n\n".join(parts)


# =============================================================================
# Small versioned distractors: older policies with different numbers
# =============================================================================

SMALL_DISTRACTORS = {
    "aurora/expense-policy-2025.txt": """
        Aurora Robotics — Travel and Expense Policy
        Version 2025.1, for the year 2025 (superseded on 1 March 2026 by the current policy).

        Meals: up to 65 USD a day while travelling; client dinners up to 100 USD per attendee.
        Hotels: up to 200 USD a night; 280 USD in New York, London, Tokyo and Zurich.
        Mileage: 0.58 USD per mile.
        Claims: within 45 days of the trip; nothing older than 120 days is paid.
        Approvals: any single expense above 400 USD needs a receipt and your manager's approval.
        Business class: flights longer than 10 hours, with a director's approval.
        """,
    "bluefin/expense-policy-2025.txt": """
        Bluefin Analytics — Expenses
        Version 2025.2, for the year 2025 (replaced by the 2026 policy on 1 January 2026).

        Meals while travelling: up to 55 USD a day. Team dinners: up to 40 USD per person, once a quarter.
        Hotels: up to 175 USD a night; 240 USD in San Francisco and New York.
        Home-office stipend: 500 USD once, plus 40 USD a month for internet.
        Claims within 60 days in Ramp; anything above 200 USD needs your manager's approval first.
        """,
    "aurora/salary-bands-2026.txt": """
        Aurora Robotics — 2026 salary bands (restricted). Version 2026.1, for the year 2026; replaced by
        the 2027 bands.

        Engineer II: 112,000 to 136,000. Senior Engineer: 139,000 to 170,000. Staff Engineer: 172,000 to
        205,000. Technician I: 55,000 to 65,000. Technician II: 66,000 to 78,000.
        The 2026 merit budget was 3.5 percent of payroll.
        """,
    "bluefin/salary-bands-2026.txt": """
        Bluefin Analytics — 2026 salary bands (restricted). Version 2026.1, for the year 2026; replaced by
        the 2027 bands.

        Data Engineer II: 120,000 to 144,000. Senior Data Engineer: 146,000 to 178,000. ML Engineer II:
        126,000 to 151,000. Senior ML Engineer: 154,000 to 188,000. Support Specialist: 60,000 to 75,000.
        The 2026 merit budget was 3.9 percent of payroll.
        """,
}


def main() -> None:
    rng = random.Random(SEED)
    out: dict[str, str] = {}
    for org in ORGS:
        for year in (2025, 2026):
            name, text = handbook(org, year)
            out[name] = text
    for path, spec in PRODUCTS.items():
        out[path] = manual(path, spec, rng)
    for path, c in CONTRACTS.items():
        out[path] = contract(path, c)
    for build in (postmortem_aurora, postmortem_bluefin):
        name, text = build(rng)
        out[name] = text
    for org in ORGS:
        name, text = release_notes(org, rng)
        out[name] = text
    for path, text in SMALL_DISTRACTORS.items():
        out[path] = dedent(text).strip()
        fact(path, "versioned", "Older version with different numbers", "Use with the current policy's question")

    for path, text in out.items():
        target = DOCS / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text.strip() + "\n", encoding="utf-8")

    write_ledger()
    words = {p: len(t.split()) for p, t in out.items()}
    for p, w in sorted(words.items(), key=lambda kv: -kv[1]):
        print(f"{w:6d}  {p}")
    print(f"{sum(words.values()):6d}  words in {len(words)} generated documents")


def write_ledger() -> None:
    lines = ["# Facts planted by kb/tools/build_long_docs.py, for writing eval questions.",
             "# kind: exact (full-text search should win), paraphrase (dense search should win),",
             "# versioned (an older or sibling version holds a different value), mirrored (the other",
             "# org's document holds a different value). 'ask' is a suggested question.", "facts:"]
    for f in LEDGER:
        lines.append(f"  - doc: {f['doc']}")
        lines.append(f"    kind: {f['kind']}")
        lines.append(f"    fact: {quote(f['fact'])}")
        lines.append(f"    ask: {quote(f['ask'])}")
    (ROOT / "facts.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


if __name__ == "__main__":
    main()
