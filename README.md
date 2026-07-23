# AI-powered-EHR-Assistant
!pip -q uninstall -y google-adk langgraph-prebuilt \
  opentelemetry-sdk opentelemetry-api \
  opentelemetry-exporter-otlp-proto-http opentelemetry-exporter-otlp-proto-common opentelemetry-proto \
  opentelemetry-exporter-gcp-logging

  # Install required libraries
!pip install -q langchain_community==0.3.27 \
              langchain==0.3.27 \
              chromadb==1.0.15 \
              pymupdf==1.26.3 \
              tiktoken==0.9.0 \
              datasets==4.0.0 \
              langchain_openai==0.3.30

              import os              # environment variables / runtime config
import re              # text normalization + lightweight pattern matching
import json            # serialize/parse tool outputs + structured data interchange
import sqlite3         # connect/query the SQLite database
import pandas as pd    # load/query CSV reference data
import numpy as np     # numeric utilities
from typing import Any, Dict, List, Optional, TypedDict, Tuple  # type hints + structured state for agent workflow

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, ToolMessage  # chat + tool observation messages
from langchain_core.prompts import ChatPromptTemplate  # consistent prompt templates
from langchain_core.tools import tool                 # define LLM-callable tools
from langchain_core.output_parsers import JsonOutputParser  # parse LLM outputs into JSON

from langchain_openai import ChatOpenAI  # OpenAI chat model wrapper

from langgraph.graph import StateGraph, END  # build the LangGraph state machine + termination node

from google.colab import drive
drive.mount('/content/drive')


import os
os.chdir('/content/drive/MyDrive/AI/WEEK1-PROMPTENGINEERING')

DB_PATH = "Project 2 - AI-powered EHR Assistant - Data Files/health_portal.db"    # complete the code to add the path to the SQLite database file

CSV_TRUSTED_SOURCES = "Project 2 - AI-powered EHR Assistant - Data Files/trusted_sources_catalog.csv"    # complete the code to add the path to the trusted_sources_catalog.csv file
CSV_LAB_EXPLAIN =  "Project 2 - AI-powered EHR Assistant - Data Files/patient_friendly_lab_explanations.csv"    # complete the code to add the path to the patient_friendly_lab_explanations.csv file
CSV_MED_EDU =  "Project 2 - AI-powered EHR Assistant - Data Files/medication_education.csv"    # complete the code to add the path to the medication_education.csv file
CSV_POLICY =  "Project 2 - AI-powered EHR Assistant - Data Files/safety_policy_rules.csv"    # complete the code to add the path to the safety_policy_rules.csv file

trusted_sources_df = pd.read_csv(CSV_TRUSTED_SOURCES)
lab_explain_df     = pd.read_csv(CSV_LAB_EXPLAIN)
med_edu_df         = pd.read_csv(CSV_MED_EDU)
policy_rules_df    = pd.read_csv(CSV_POLICY)

print("trusted_sources rows:", len(trusted_sources_df))
print("lab explanations rows:", len(lab_explain_df))
print("med education rows:", len(med_edu_df))
print("policy rules rows:", len(policy_rules_df))

con = sqlite3.connect(DB_PATH)
con.row_factory = sqlite3.Row  # nicer dict-like rows
cur = con.cursor()

# List tables
tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
print("Tables:", [t["name"] for t in tables])

# Check row counts
for t in ["patients","encounters","clinical_notes","labs","medications","allergies"]:
    c = cur.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"]
    print(f"{t:15s} {c}")

    policy_rules_df.head()

    # Load the JSON file and extract values
file_name = "/content/drive/MyDrive/AI/config.json"                                                       # Name of the configuration file
with open(file_name, 'r') as file:                                              # Open the config file in read mode
    config = json.load(file)                                                    # Load the JSON content as a dictionary
    OPENAI_API_KEY = config.get("OPENAI_API_KEY")                                             # Extract the API key from the config
    OPENAI_API_BASE = config.get("OPEN_API_BASE")                             # Extract the OpenAI base URL from the config

# Store API credentials in environment variables
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY                                          # Set API key as environment variable
os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE                                 # Set API base URL as environment variable

# Initialize OpenAI client
#client = OpenAI()                                                               # Create an instance of the OpenAI client


# define the generator llm
# uncomment one of the following code snippets to choose the LLM for output generation
# GEN_MODEL = "gpt-4o-mini"
# GEN_MODEL = "gpt-4o"

# define the validator llm
# uncomment one of the following code snippets to choose the LLM for output generation
# VAL_MODEL = "gpt-4o-mini"
# VAL_MODEL = "gpt-4o"

# set up the generator llm
llm_gen = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0    # complete the code to define the temperature for the LLM
)

# set up the validator llm
llm_val = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0    # complete the code to define the temperature for the LLM
)

def sql_query(q: str, params: tuple = ()):
    """Execute a parameterized SQL query on the SQLite cursor and return rows as dictionaries.

    Inputs:
        q (str): SQL query string (use ? placeholders for parameters).
        params (tuple): Parameters to bind to the query placeholders.

    Output:
        List[Dict[str, Any]]: Query results where each row is converted to a dict
        (column_name -> value). Returns an empty list if no rows match.
    """
    rows = cur.execute(q, params).fetchall()  # run query + fetch all result rows
    return [dict(r) for r in rows]            # convert sqlite3.Row objects to plain dicts

    def norm_text(s: str) -> str:
    """Normalize text for consistent matching (lowercase + trimmed + whitespace-collapsed).

    Inputs:
        s (str): Any input text (will be cast to string).

    Output:
        str: Normalized text with:
            - leading/trailing whitespace removed
            - converted to lowercase
            - internal whitespace collapsed to single spaces
    """
    return re.sub(r"\s+", " ", str(s).strip().lower())  # standardize spacing/case for comparisons

    def to_json(obj: Any) -> str:
    """Convert a Python object into a JSON string safely.

    Inputs:
        obj (Any): Any Python object (dict/list/str/etc.). If the object contains
            non-JSON-serializable values (e.g., datetime, numpy types), they are
            converted to strings via default=str.

    Output:
        str: JSON-formatted string (UTF-8 friendly, ensure_ascii=False).
    """
    return json.dumps(obj, ensure_ascii=False, default=str)  # robust serialization for tool outputs/logging

    @tool("get_patient_profile")
def get_patient_profile(patient_id: str) -> str:
    """
    Retrieve a patient's basic profile from the SQLite `patients` table for personalization.

    Args:
        patient_id (str): Unique patient identifier (e.g., "P001") used to look up the patient record.

    Returns:
        str: A JSON string containing the patient profile fields:
             patient_id, first_name, last_name, birth_year, sex_at_birth,
             preferred_language, health_literacy_level, timezone, created_at.
             If the patient is not found, returns a JSON string like:
             {"error": "Patient <patient_id> not found"}.
    """
    rows = sql_query("""
        SELECT patient_id, first_name, last_name, birth_year, sex_at_birth,
               preferred_language, health_literacy_level, timezone, created_at
        FROM patients
        WHERE patient_id = ?
    """, (patient_id,))
    return to_json(rows[0] if rows else {"error": f"Patient {patient_id} not found"})


    @tool("list_patient_encounters")
def list_patient_encounters(patient_id: str, limit: int = 5) -> str:
    """
    Fetch a patient's most recent encounters from the SQLite `encounters` table.

    This tool is used to provide visit context (e.g., last appointment type, reason for visit,
    follow-up instructions) for summarization and navigation-style answers.

    Args:
        patient_id (str): Unique patient identifier (e.g., "P003").
        limit (int, optional): Maximum number of encounters to return. For safety and
            consistency, this value can be capped (between 3-7). Defaults to 5.

    Returns:
        str: A JSON string containing a list of encounter objects, each including:
             encounter_id, encounter_date, encounter_type, reason_for_visit, diagnosis_summary,
             provider_specialty, followup_instructions, care_team_contact.
             Returns "[]" (JSON empty list) if no encounters are found.
    """
    max_val =2    # complete the code to define the maximum number of encounters to return
    try:
        limit = max(1, min(int(limit), max_val))
    except Exception:
        limit = max_val

    rows = sql_query("""
        SELECT encounter_id, encounter_date, encounter_type, reason_for_visit,
               diagnosis_summary, provider_specialty, followup_instructions, care_team_contact
        FROM encounters
        WHERE patient_id = ?
        ORDER BY encounter_date DESC
        LIMIT ?
    """, (patient_id, limit))
    return to_json(rows)


@tool("get_recent_clinical_note")
def get_recent_clinical_note(patient_id: str, note_type: str = "visit_note") -> str:
    """
    Retrieve the most recent clinical note for a patient from the SQLite `clinical_notes` table.

    This tool supports note summarization and “what happened at my last visit?” questions by
    pulling the latest note text for a specified note category.

    Args:
        patient_id (str): Unique patient identifier (e.g., "P007").
        note_type (str, optional): Type of note to retrieve. Allowed values are:
            - "visit_note"
            - "discharge_summary"
            Any other value is defaulted to "visit_note".

    Returns:
        str: A JSON string containing a single note object with fields:
             note_id, encounter_id, patient_id, note_type, note_text, created_at, author_role.
             If no matching note exists, returns a JSON string like:
             {"error": "No <note_type> found for patient <patient_id>"}.
    """
    if note_type not in ("visit_note", "discharge_summary"):
        note_type = "visit_note"

    rows = sql_query("""
        SELECT note_id, encounter_id, patient_id, note_type, note_text, created_at, author_role
        FROM clinical_notes
        WHERE patient_id = ? AND note_type = ?
        ORDER BY created_at DESC
        LIMIT 1
    """, (patient_id, note_type))
    return to_json(rows[0] if rows else {"error": f"No {note_type} found for patient {patient_id}"})


    @tool("get_clinical_notes_for_encounter")
def get_clinical_notes_for_encounter(encounter_id: str) -> str:
    """
    Retrieve all clinical notes linked to a specific encounter from the SQLite `clinical_notes` table.

    This tool is useful when the agent first identifies the relevant encounter (e.g., the most recent
    visit from `list_patient_encounters`) and then needs to fetch all associated note types (visit note,
    discharge summary) for a complete, encounter-scoped summary.

    Args:
        encounter_id (str): Unique encounter identifier to fetch notes for.

    Returns:
        str: A JSON string containing a list of note objects. Each note includes:
             note_id, encounter_id, patient_id, note_type, note_text, created_at, author_role.
             Returns "[]" (JSON empty list) if no notes are found for the encounter.
    """
    rows = sql_query("""
        SELECT note_id, encounter_id, patient_id, note_type, note_text, created_at, author_role
        FROM clinical_notes
        WHERE encounter_id = ?
        ORDER BY created_at ASC
    """, (encounter_id,))
    return to_json(rows)


    @tool("get_labs")
def get_labs(patient_id: str, test_name: Optional[str] = None, limit: int = 10) -> str:
    """
    Fetch recent laboratory results for a patient from the SQLite `labs` table.

    This tool supports lab-explanation use cases by retrieving the most recent lab records,
    optionally filtered to a specific lab test name (case-insensitive exact match).

    Args:
        patient_id (str): Unique patient identifier (e.g., "P004").
        test_name (Optional[str], optional): If provided, restricts results to rows where
            `test_name` matches (case-insensitive). If None, returns a mix of recent labs.
        limit (int, optional): Maximum number of rows to return. For safety and prompt
            size control, the value can be capped (between 7-10). Defaults to 10.

    Returns:
        str: A JSON string containing a list of lab result objects, each including:
             lab_result_id, ordered_date, result_date, loinc_code, test_name, value_numeric,
             value_text, unit, ref_range_low, ref_range_high, flag, lab_source.
             Returns "[]" (JSON empty list) if no results are found.
    """
    max_val =2   # complete the code to define the maximum number of rows to return
    try:
        limit = max(1, min(int(limit), max_val))
    except Exception:
        limit = max_val

    if test_name:
        rows = sql_query("""
            SELECT lab_result_id, ordered_date, result_date, loinc_code, test_name,
                   value_numeric, value_text, unit, ref_range_low, ref_range_high, flag, lab_source
            FROM labs
            WHERE patient_id = ? AND lower(test_name) = lower(?)
            ORDER BY result_date DESC
            LIMIT ?
        """, (patient_id, test_name, limit))
    else:
        rows = sql_query("""
            SELECT lab_result_id, ordered_date, result_date, loinc_code, test_name,
                   value_numeric, value_text, unit, ref_range_low, ref_range_high, flag, lab_source
            FROM labs
            WHERE patient_id = ?
            ORDER BY result_date DESC
            LIMIT ?
        """, (patient_id, limit))

    return to_json(rows)


    @tool("get_medications")
def get_medications(patient_id: str, status: str = "active") -> str:
    """
    Retrieve a patient's medication list from the SQLite `medications` table.

    This tool supports medication education and reconciliation-style questions by returning
    the patient's recorded medications along with basic directions (dose/route/frequency),
    status (active/stopped), and a high-level indication when available.

    Args:
        patient_id (str): Unique patient identifier (e.g., "P002").
        status (str, optional): Filter for medication status:
            - "active": only active meds
            - "stopped": only stopped meds
            - "all": both active and stopped
            Any other value defaults to "active".

    Returns:
        str: A JSON string containing a list of medication objects, each including:
             med_id, rxnorm_code, med_name, dose, route, frequency, start_date, end_date,
             status, indication, prescriber_specialty.
             Returns "[]" (JSON empty list) if no medications match the filter.
    """
    if status not in ("active", "stopped", "all"):
        status = "active"

    if status == "all":
        rows = sql_query("""
            SELECT med_id, rxnorm_code, med_name, dose, route, frequency,
                   start_date, end_date, status, indication, prescriber_specialty
            FROM medications
            WHERE patient_id = ?
            ORDER BY status DESC, start_date DESC
        """, (patient_id,))
    else:
        rows = sql_query("""
            SELECT med_id, rxnorm_code, med_name, dose, route, frequency,
                   start_date, end_date, status, indication, prescriber_specialty
            FROM medications
            WHERE patient_id = ? AND status = ?
            ORDER BY start_date DESC
        """, (patient_id, status))

    return to_json(rows)

    @tool("get_allergies")
def get_allergies(patient_id: str) -> str:
    """
    Retrieve a patient's recorded allergies from the SQLite `allergies` table.

    This tool supports safety-aware responses (e.g., antibiotic/allergy concerns) by returning
    the substances, reactions, and severity recorded in the patient record.

    Args:
        patient_id (str): Unique patient identifier (e.g., "P005").

    Returns:
        str: A JSON string containing a list of allergy objects, each including:
             allergy_id, substance, reaction, severity, recorded_date.
             Returns "[]" (JSON empty list) if no allergies are recorded for the patient.
    """
    rows = sql_query("""
        SELECT allergy_id, substance, reaction, severity, recorded_date
        FROM allergies
        WHERE patient_id = ?
        ORDER BY recorded_date DESC
    """, (patient_id,))
    return to_json(rows)

    @tool("lookup_lab_education")
def lookup_lab_education(test_name: str) -> str:
    """
    Retrieve patient-friendly education content for a lab test from `patient_friendly_lab_explanations.csv`.

    This tool provides plain-language explanations and safe guidance for a lab test, along with
    citation metadata (e.g., citation_url and source_id) that the agent can reference to ground
    general medical information.

    Args:
        test_name (str): Lab test name to look up (e.g., "Hemoglobin A1c"). Matching is done using
            a normalized exact match first, then a fallback substring match.

    Returns:
        str: A JSON string containing the education row fields (e.g., test_name_normalized,
             plain_language_summary, why_it_matters, common_reasons_high/low, when_to_contact_clinician,
             citation_url, source_id). If no match is found, returns:
             {"error": "No lab education found for '<test_name>'"}.
    """
    tn = norm_text(test_name)
    df = lab_explain_df.copy()
    df["_k"] = df["test_name_normalized"].astype(str).map(norm_text)

    hit = df[df["_k"] == tn]
    if hit.empty:
        hit = df[df["_k"].str.contains(tn, na=False)]

    if hit.empty:
        return to_json({"error": f"No lab education found for '{test_name}'"})

    row = hit.iloc[0].drop(labels=["_k"]).to_dict()
    return to_json(row)


    @tool("lookup_medication_education")
def lookup_medication_education(med_name: str) -> str:
    """
    Retrieve patient-friendly education content for a medication from `medication_education.csv`.

    This tool provides general medication education (what it is for, common/serious side effects,
    general interaction warnings) plus citation metadata (citation_url, source_id) to support
    grounded, non-prescriptive answers.

    Args:
        med_name (str): Medication name to look up (e.g., "Metformin"). Matching is done using
            a normalized exact match first, then a fallback substring match.

    Returns:
        str: A JSON string containing the education row fields (e.g., med_name_normalized,
             drug_class, what_it_is_for, common_side_effects, serious_side_effects_red_flags,
             interaction_warnings_general, citation_url, source_id). If no match is found, returns:
             {"error": "No medication education found for '<med_name>'"}.
    """
    mn = norm_text(med_name)
    df = med_edu_df.copy()
    df["_k"] = df["med_name_normalized"].astype(str).map(norm_text)

    hit = df[df["_k"] == mn]
    if hit.empty:
        hit = df[df["_k"].str.contains(mn, na=False)]

    if hit.empty:
        return to_json({"error": f"No medication education found for '{med_name}'"})

    row = hit.iloc[0].drop(labels=["_k"]).to_dict()
    return to_json(row)

    @tool("lookup_trusted_source")
def lookup_trusted_source(source_id: str) -> str:
    """
    Retrieve metadata for an approved medical information source from `trusted_sources_catalog.csv`.

    This tool supports citation grounding by mapping a `source_id` (stored in lab/med education rows)
    to a human-readable source name and URL.

    Args:
        source_id (str): Identifier for the trusted source (e.g., "S01").

    Returns:
        str: A JSON string containing the source metadata fields (e.g., source_id, source_name,
             base_url, content_type, license_notes, is_allowed). If the source_id is not found,
             returns: {"error": "source_id <source_id> not found"}.
    """
    hit = trusted_sources_df[trusted_sources_df["source_id"] == source_id]
    if hit.empty:
        return to_json({"error": f"source_id {source_id} not found"})
    return to_json(hit.iloc[0].to_dict())


    @tool("policy_route")
def policy_route(user_text: str) -> str:
    """
    Apply safety guardrails to a user query using `safety_policy_rules.csv` and return a routing decision.

    This tool implements a lightweight, deterministic policy layer that helps the agent decide whether to:
    - answer normally ("answer"),
    - refuse unsafe requests such as diagnosis/dose changes ("refuse"),
    - escalate urgent symptom descriptions ("escalate_emergency" or "escalate_clinician").

    Matching approach (MVP):
    - Normalizes the user text and checks whether each rule's `pattern_or_topic` appears as a substring.
    - If multiple rules match, selects the highest-priority action using:
      escalate_emergency > escalate_clinician > refuse > answer.

    Args:
        user_text (str): The raw user input/query to evaluate.

    Returns:
        str: A JSON string with keys:
            - action: "answer" | "refuse" | "escalate_emergency" | "escalate_clinician"
            - rule_id: matched rule identifier (if any)
            - template: standard response template for refusal/escalation (if any)
            - matched_rules: list of all matched rules with their actions/topics
        If no rules match, returns:
            {"action": "answer", "matched_rules": []}
    """
    text = norm_text(user_text)
    rules = policy_rules_df.to_dict(orient="records")

    matches = []
    for r in rules:
        topic = norm_text(r.get("pattern_or_topic", ""))
        if topic and topic in text:
            matches.append(r)

    priority = {"escalate_emergency": 3, "escalate_clinician": 2, "refuse": 1, "answer": 0}

    if not matches:
        return to_json({"action": "answer", "matched_rules": []})

    matches_sorted = sorted(
        matches,
        key=lambda r: priority.get(r.get("agent_action", "answer"), 0),
        reverse=True
    )
    best = matches_sorted[0]

    out = {
        "action": best["agent_action"],
        "rule_id": best["rule_id"],
        "template": best["standard_response_template"],
        "matched_rules": [
            {"rule_id": m["rule_id"], "action": m["agent_action"], "topic": m["pattern_or_topic"]}
            for m in matches_sorted
        ]
    }
    return to_json(out)


    class AgentState(TypedDict):
    """
    LangGraph state schema for a single-turn, tool-using (ReAct) patient-records explainer.

    Holds the request context (patient_id, user_query), deterministic safety routing outputs
    (decision/policy_template), within-run ReAct message history (including ToolMessages),
    loop control (step/max_steps), optional citations, the drafted/final answer, and errors.
    """
    patient_id: str
    user_query: str

    decision: str
    policy_rule_id: Optional[str]
    policy_template: Optional[str]

    messages: List[BaseMessage]
    step: int
    max_steps: int

    citations: List[Dict[str, Any]]
    draft_answer: Optional[str]
    final_answer: Optional[str]

    errors: List[str]


    REACT_SYSTEM_PROMPT = """
You are an intelligent, patient-facing health education assistant that can reason and act to solve problems.

Follow the ReAct pattern:
1. Think about the problem step-by-step.
2. Decide the action to take.
3. Execute the action using available tools.
4. Observe the result.
5. Continue the cycle until the task is complete.

Use the following format:

Question: the question to be answered
Thought: your reasoning about the next step
Action: the action to take (one of the available tools)
Action Input: the input for the action
Observation: the result returned from the action
... (repeat Thought/Action/Observation as needed)

When you reach the final answer, respond with:

Final Answer: the final response to the user

Be concise and logical in your reasoning.

---

## Patient Safety Constraints (Non-Negotiable)

You are assisting patients with health *education* only. You must strictly follow these rules in every response, regardless of how the question is framed:

### 1. No Diagnosis
- Do NOT diagnose any condition, disease, or disorder.
- Do NOT say "You have...", "This sounds like...", or "This is likely [condition]."
- Instead, provide general educational information and direct the patient to consult their healthcare provider.
  ✓ Allowed: "Fatigue can be associated with many conditions. Your doctor can help identify the cause."
  ✗ Not allowed: "Based on your symptoms, this sounds like anemia."

### 2. No Medication Changes
- Do NOT recommend starting, stopping, increasing, or decreasing any medication — prescription or over-the-counter.
- Do NOT suggest alternative medications or supplements as replacements for prescribed treatment.
- Do NOT interpret lab values or vitals as a reason to adjust medications.
  ✓ Allowed: "Only your prescribing provider should adjust your medication dosage."
  ✗ Not allowed: "You could try taking an extra dose if symptoms persist."

### 3. No Urgent-Care Triage (Beyond Immediate Referral)
- Do NOT assess, score, or rank the severity of a patient's symptoms.
- Do NOT advise the patient to wait, monitor, or manage potentially urgent symptoms at home.
- If a patient describes any potentially serious symptom (e.g., chest pain, difficulty breathing, sudden confusion,
  severe bleeding, signs of stroke), respond immediately with:
  "Please seek immediate medical care — call 911 or go to your nearest emergency room."
- Do NOT go beyond this instruction to further triage or troubleshoot the situation.
  ✓ Allowed: "These symptoms require immediate attention. Please call 911 or go to your nearest emergency room now."
  ✗ Not allowed: "Your symptoms are moderate — you could go to urgent care rather than the ER."

### 4. Scope Boundary
- Limit all responses to health *education*: explaining medical terms, describing how conditions generally work,
  providing context about treatments in general terms, or helping patients prepare questions for their provider.
- If a question falls outside this scope, clearly state:
  "This is outside what I'm able to help with. Please speak directly with your healthcare provider."

### 5. Disclaimer
- Where appropriate, remind the patient:
  "This information is for educational purposes only and is not a substitute for professional medical advice,
  diagnosis, or treatment."
"""


def init_state(patient_id: str, user_query: str, max_steps: int = 5) -> AgentState:
    return {
        "patient_id": patient_id,
        "user_query": user_query,

        "decision": "answer",
        "policy_rule_id": None,
        "policy_template": None,

        "messages": [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            HumanMessage(content=f"patient_id={patient_id}\nUser question: {user_query}")
        ],
        "step": 0,
        "max_steps": max_steps,

        "citations": [],
        "draft_answer": None,
        "final_answer": None,
        "errors": [],
    }

    TOOLS = [
    get_patient_profile,
    list_patient_encounters,
    get_recent_clinical_note,
    get_clinical_notes_for_encounter,
    get_labs,
    get_medications,
    get_allergies,
    lookup_lab_education,
    lookup_medication_education,
    lookup_trusted_source,
]

# enable the LLM to call the provided tools via native tool/function-calling
llm_gen_tools = llm_gen.bind_tools(TOOLS)    # complete the code to define the list of tools to be provided to the LLM for calling

def policy_gate_node(state: AgentState) -> Dict[str, Any]:
    """
    Run the deterministic policy router on the user's query and write the resulting safety decision into state.

    Inputs:
        state (AgentState): Current graph state containing at least `user_query` (and existing `errors` if any).

    Outputs:
        Dict[str, Any]: State updates including:
          - decision: "answer" | "refuse" | "escalate_emergency" | "escalate_clinician"
          - policy_rule_id: matched rule id (or None)
          - policy_template: standardized refusal/escalation message (or None)
        On failure, returns an update to `errors` with a descriptive message.
    """
    try:
        route = json.loads(policy_route.invoke({"user_text": state["user_query"]}))
        return {
            "decision": route.get("action", "answer"),
            "policy_rule_id": route.get("rule_id"),
            "policy_template": route.get("template"),
        }
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"policy_gate failed: {e}"]}

        def agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Run one ReAct "agent" step using the tool-bound LLM.

    The node:
    - short-circuits for emergency escalation by emitting a draft answer from the policy template,
    - enforces a max-step hard stop to prevent infinite tool loops,
    - otherwise calls the LLM with the current message history and appends the AI message.

    Inputs:
        state (AgentState): Current state containing `messages`, `step`, `max_steps`, and policy fields.

    Outputs:
        Dict[str, Any]: State updates that always include `messages` and `step`, and optionally:
          - `draft_answer` when the LLM returns a final text response (no tool calls) or when hard-stopped.
        If the LLM returns tool calls, this node does not set `draft_answer`, allowing the graph to route
        to the tool execution node.
    """
    # Emergency short-circuit: finalize will enforce template; still produce a draft to stop loop
    if state.get("decision") == "escalate_emergency" and state.get("policy_template"):
        return {
            "draft_answer": state["policy_template"],
            "messages": state["messages"],  # still write something explicit
            "step": state["step"],
        }

    # Hard-stop
    if state["step"] >= state["max_steps"]:
        forced = (
            "I may not have enough information yet to answer confidently. "
            "I can explain what your records show or share general information. "
            "For medical advice (especially medication changes) or urgent concerns, please contact your clinician."
        )
        return {
            "draft_answer": forced,
            "messages": state["messages"],
            "step": state["step"],
        }

    # Invoke LLM with tools
    ai_msg = llm_gen_tools.invoke(state["messages"])

    # Explicitly build updated messages + step (no in-place mutation reliance)
    new_messages = list(state["messages"]) + [ai_msg]
    new_step = state["step"] + 1

    # If no tool calls, treat as draft answer and stop looping
    if not getattr(ai_msg, "tool_calls", None):
        return {
            "messages": new_messages,
            "step": new_step,
            "draft_answer": ai_msg.content,
        }

    # Tool calls exist: continue loop, but MUST still return an update
    return {
        "messages": new_messages,
        "step": new_step,
    }

    # tools that must be forced to use the logged-in patient_id (prevents cross-patient access)
PATIENT_SCOPED_TOOLNAMES = {
    "get_patient_profile",
    "list_patient_encounters",
    "get_recent_clinical_note",
    "get_labs",
    "get_medications",
    "get_allergies",
}


def _cap_limit(args: Dict[str, Any], max_limit: int = 10) -> Dict[str, Any]:
    """
    Cap a tool call's 'limit' argument to control data volume and prompt size.

    Args:
        args (Dict[str, Any]): Tool argument dictionary (may include a 'limit' key).
        max_limit (int, optional): Maximum allowed limit value. Defaults to 10.

    Returns:
        Dict[str, Any]: A copied args dictionary with 'limit' coerced to int and capped to max_limit
        when present; otherwise returns args unchanged.
    """
    out = dict(args)  # copy args so we don't mutate the original dict
    if "limit" in out and out["limit"] is not None:
        try:
            out["limit"] = min(int(out["limit"]), max_limit)  # cap numeric limit
        except Exception:
            out["limit"] = max_limit  # fallback if limit isn't parseable
    return out


    def tool_exec_node(state: AgentState) -> Dict[str, Any]:
    """
    Execute tool calls produced by the tool-enabled LLM and append results back into the message trace.

    This node:
    - reads tool calls from the last AI message,
    - enforces patient scoping by overwriting patient_id for sensitive tools,
    - caps limit parameters for controlled retrieval,
    - executes each tool from the approved tool map,
    - appends each tool output as a ToolMessage observation,
    - optionally extracts citations from lab/med education tool outputs.

    Inputs:
        state (AgentState): Current state containing at least:
            - patient_id: used to enforce patient scoping
            - messages: last message may contain tool_calls
            - citations/errors: existing lists to extend

    Outputs:
        Dict[str, Any]: State updates containing:
            - messages: updated message list including ToolMessages (tool observations)
            - citations: updated list with extracted citation metadata (if any)
            - errors: updated list with any tool execution / blocking errors
    """
    last = state["messages"][-1]  # last AI message should contain tool_calls
    tool_calls = getattr(last, "tool_calls", []) or []  # tool call specs from LLM
    tool_map = {t.name: t for t in TOOLS}  # whitelist: tool name -> tool callable

    new_messages = list(state["messages"])                 # copy messages (avoid in-place mutation)
    new_citations = list(state.get("citations", []))       # copy citations
    new_errors = list(state.get("errors", []))             # copy errors

    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("args", {}) or {}
        tool_id = tc.get("id")  # required so ToolMessage can be associated to the correct call

        # Block any tool not in our allowlist (security)
        if name not in tool_map:
            new_errors.append(f"Blocked unknown tool call: {name}")
            new_messages.append(
                ToolMessage(content=f"Blocked unknown tool: {name}", name=name or "unknown", tool_call_id=tool_id)
            )
            continue

        # Critical: prevent cross-patient data access by enforcing patient_id
        if name in PATIENT_SCOPED_TOOLNAMES:
            args["patient_id"] = state["patient_id"]

        # Cap retrieval volume (limits prompt size and reduces accidental overexposure)
        if name in {"get_labs", "list_patient_encounters"}:
            args = _cap_limit(args, max_limit=10)

        try:
            # Execute tool (returns JSON string in our implementation)
            result_str = tool_map[name].invoke(args)
            if not isinstance(result_str, str):
                result_str = json.dumps(result_str, ensure_ascii=False, default=str)

            # Optional: extract citation metadata from education tools
            if name in {"lookup_lab_education", "lookup_medication_education"}:
                try:
                    data = json.loads(result_str)
                    if isinstance(data, dict) and "error" not in data:
                        cit = {
                            "source_id": data.get("source_id"),
                            "citation_url": data.get("citation_url"),
                            "used_for": "lab education" if name == "lookup_lab_education" else "med education",
                        }
                        if cit.get("source_id") or cit.get("citation_url"):
                            new_citations.append(cit)
                except Exception:
                    pass  # if parsing fails, skip citation extraction silently (MVP-friendly)

            # Append tool observation for the agent to use in the next step (ReAct "Observation")
            new_messages.append(ToolMessage(content=result_str, name=name, tool_call_id=tool_id))

        except Exception as e:
            err = f"{name} failed: {e}"
            new_errors.append(err)
            new_messages.append(ToolMessage(content=err, name=name, tool_call_id=tool_id))

    return {
        "messages": new_messages,
        "citations": new_citations,
        "errors": new_errors,
    }  # explicit updates required by LangGraph


    def should_continue(state: AgentState) -> str:
    """
    Determine the next step in the ReAct loop based on the latest state.

    Inputs:
        state (AgentState): Current state containing:
          - draft_answer (optional): set when the agent has produced a final response
          - messages: includes the most recent AI message which may contain tool_calls

    Returns:
        str: The next node label for LangGraph routing:
          - "final" if a draft_answer exists (stop looping and finalize)
          - "tool" if the last AI message contains tool_calls (execute tools next)
          - "agent" otherwise (continue by calling the agent again)
    """
    if state.get("draft_answer"):
        return "final"  # agent has produced an answer; exit loop

    last = state["messages"][-1]  # inspect the most recent AI message
    if getattr(last, "tool_calls", None):
        return "tool"  # tools requested; execute them

    return "final"  # replaced from agent to Final no answer yet and no tools requested; ask agent again

    def final_policy_override_node(state: AgentState) -> Dict[str, Any]:
    """
    Apply the final, deterministic safety override to select the response returned to the user.

    Inputs:
        state (AgentState): Current state containing:
          - decision: policy decision (answer/refuse/escalate_*)
          - policy_template: standardized refusal/escalation message (if applicable)
          - draft_answer: agent-generated answer (if applicable)

    Outputs:
        Dict[str, Any]: State update with:
          - final_answer: the policy template if decision requires it; otherwise the draft_answer,
            or a conservative fallback if no draft exists.
    """
    decision = state.get("decision", "answer")
    template = state.get("policy_template")

    # If policy requires refusal/escalation, override any LLM output
    if decision in ("escalate_emergency", "refuse", "escalate_clinician") and template:
        return {"final_answer": template}

    # Otherwise return the model's draft answer (or a fallback)
    return {"final_answer": state.get("draft_answer") or "I’m not sure how to answer that yet."}


    # Build a LangGraph state machine using AgentState as the shared state object
graph = StateGraph(AgentState)

# Register nodes: policy gate -> ReAct agent -> tool execution -> final policy override
graph.add_node("policy", policy_gate_node)                 # deterministic safety routing
graph.add_node("agent", agent_node)                        # LLM step (may request tools or answer)
graph.add_node("tool", tool_exec_node)                     # executes requested tools + appends observations
graph.add_node("final", final_policy_override_node)        # deterministic final safety enforcement
graph.set_entry_point("policy")                            # start every run with the safety gate

# If emergency escalation is detected, short-circuit directly to final; otherwise enter the ReAct loop
graph.add_conditional_edges(
    "policy",
    lambda s: "final" if s.get("decision") == "escalate_emergency" else "agent",
    {"final": "final", "agent": "agent"}
)

# ReAct loop routing: if tools requested -> tool node; if answer ready -> final; else keep calling agent
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tool": "tool", "final": "final"}
)

graph.add_edge("tool", "agent")                            # after tool observations, return control to agent
graph.add_edge("final", END)                               # end the workflow after final answer is set

agent_app = graph.compile()                                # compile the graph into an executable app


# render and display the LangGraph workflow diagram in the notebook
display(Image(agent_app.get_graph().draw_mermaid_png()))    # complete the code to define the LangGraph workflow to visualize


pid = "P001"    # complete the code to define the patient ID
query = "What does my Hemoglobin A1c result mean?"    # complete the code to define the patient query

state = init_state(pid, query, max_steps=5)    # complete the code to define the number of iterations for the ReAct loop
out = agent_app.invoke(state)

print(out["final_answer"])


print("\n=== MESSAGE TRACE ===")
for i, m in enumerate(out["messages"]):
    print(f"\n[{i}] {type(m).__name__}")
    # Print tool call info if present
    if hasattr(m, "tool_calls") and m.tool_calls:
        print("tool_calls:", m.tool_calls)
    content = getattr(m, "content", "")
    if isinstance(content, str):
        print(content[:800])  # adjust length if needed


        pid = "P002"    # complete the code to define the patient ID
query = "What is atorvastatin used for and what are common side effects?"    # complete the code to define the patient query

state = init_state(pid, query, max_steps=2)    # complete the code to define the number of iterations for the ReAct loop
out = agent_app.invoke(state)

print(out["final_answer"])


print("\n=== MESSAGE TRACE ===")
for i, m in enumerate(out["messages"]):
    print(f"\n[{i}] {type(m).__name__}")
    # Print tool call info if present
    if hasattr(m, "tool_calls") and m.tool_calls:
        print("tool_calls:", m.tool_calls)
    content = getattr(m, "content", "")
    if isinstance(content, str):
        print(content[:800])  # adjust length if needed

        pid = "P003"    # complete the code to define the patient ID
query = "Can you summarize my most recent visit note and list the follow up instruction"    # complete the code to define the patient query

state = init_state(pid, query, max_steps=2)    # complete the code to define the number of iterations for the ReAct loop
out = agent_app.invoke(state)

print(out["final_answer"])

print("\n=== MESSAGE TRACE ===")
for i, m in enumerate(out["messages"]):
    print(f"\n[{i}] {type(m).__name__}")
    # Print tool call info if present
    if hasattr(m, "tool_calls") and m.tool_calls:
        print("tool_calls:", m.tool_calls)
    content = getattr(m, "content", "")
    if isinstance(content, str):
        print(content[:800])  # adjust length if needed

        pid = "P005"    # complete the code to define the patient ID
query = "My creatinie is high. should I stop my lisinopril ?"    # complete the code to define the patient query

state = init_state(pid, query, max_steps=5)    # complete the code to define the number of iterations for the ReAct loop
out = agent_app.invoke(state)

print(out["final_answer"])


print("\n=== MESSAGE TRACE ===")
for i, m in enumerate(out["messages"]):
    print(f"\n[{i}] {type(m).__name__}")
    # Print tool call info if present
    if hasattr(m, "tool_calls") and m.tool_calls:
        print("tool_calls:", m.tool_calls)
    content = getattr(m, "content", "")
    if isinstance(content, str):
        print(content[:800])  # adjust length if needed

        ...

        
    

from IPython.display import Image, display  # visualize the graph / notebook-friendly display

