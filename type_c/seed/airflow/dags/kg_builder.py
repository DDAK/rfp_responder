"""
kg_builder.py  –  duplicate‑safe KG loader WITH case_id propagation
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import psycopg2, json, os, decimal, hashlib
from neo4j import GraphDatabase, exceptions

PG   = dict(host="postgres", dbname="demo", user="demo", password="demo1234")
NEO  = GraphDatabase.driver("bolt://neo4j:7687",
          auth=("neo4j", os.getenv("NEO4J_PASSWORD", "demo1234")))

# Updated constraints to include new entity types from all document types
CONSTRAINTS = [
    # Existing constraints
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person)        REQUIRE p.name_lc IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Org)           REQUIRE o.name_lc IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (pr:Program)      REQUIRE pr.name_lc IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:PolicySection) REQUIRE s.id        IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Rule)          REQUIRE r.code      IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Address)       REQUIRE a.full_lc   IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Income)        REQUIRE i.id        IS UNIQUE",
    # Document entities
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Case)          REQUIRE c.id        IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Obligation)    REQUIRE o.id        IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Share)         REQUIRE s.id        IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Proportion)    REQUIRE p.id        IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Adjustment)    REQUIRE a.id        IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Payment)       REQUIRE p.id        IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Premium)       REQUIRE p.id        IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Costs)         REQUIRE c.id        IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:DocumentType)  REQUIRE d.name_lc   IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Date)          REQUIRE d.value     IS UNIQUE",
    # Person entities
    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Employee)      REQUIRE e.id        IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Contact)       REQUIRE c.id        IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Deductions)    REQUIRE d.id        IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (y:YTD_Income)    REQUIRE y.id        IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Leave)         REQUIRE l.id        IS UNIQUE",
    # Contact entities
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:SSN)           REQUIRE s.value     IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Phone)         REQUIRE p.number    IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Count)         REQUIRE c.id        IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Age)           REQUIRE a.id        IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Custody)       REQUIRE c.id        IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (b:BANK_ACCOUNT)  REQUIRE b.id        IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:CHECKING)      REQUIRE c.id        IS UNIQUE"
]

# ── normalisation helpers ────────────────────────────────────────
def canon(v: str) -> str:
    # Handle case where v might not be a string
    if not isinstance(v, str):
        return str(v).lower()
    return " ".join(v.strip().lower().split())

def label_from(t):   # maps triple types to node labels
    return {
        # Original entities
        "PERSON": "Person", "ORG": "Org", "PROGRAM": "Program",
        "POLICY_SECTION": "PolicySection", "ADDRESS": "Address",
        "INCOME": "Income", "CASE": "Case",

        # Document entities
        "OBLIGATION": "Obligation", "SHARE": "Share", "PROPORTION": "Proportion",
        "ADJUSTMENT": "Adjustment", "PAYMENT": "Payment", "PREMIUM": "Premium",
        "COSTS": "Costs", "DATE": "Date",
        "DOCUMENT_TYPE": "DocumentType",

        # Person entities
        "EMPLOYEE": "Employee", "CONTACT": "Contact", "DEDUCTIONS": "Deductions",
        "YTD_INCOME": "YTD_Income", "LEAVE": "Leave",

        # Contact entities
        "SSN": "SSN", "PHONE": "Phone", "COUNT": "Count", "AGE": "Age",
        "CUSTODY": "Custody", "BANK_ACCOUNT": "BankAccount", "CHECKING": "Checking",

        # Default
        "THING": "Thing"
    }.get(t, "Thing")

def key(label):
    # Updated key function to handle all entity types
    if label in ["Person", "Org", "Program", "DocumentType"]:
        return "name_lc"
    elif label == "Address":
        return "full_lc"
    elif label == "Date":
        return "value"
    elif label == "SSN":
        return "value"
    elif label == "Phone":
        return "number"
    else:
        # All other entities use id as key
        return "id"

def disp(label):
    # Updated display function to handle all entity types
    if label in ["Person", "Org", "Program", "DocumentType"]:
        return "name"
    elif label == "Address":
        return "full"
    elif label == "Date" or label == "SSN":
        return "value"
    elif label == "Phone":
        return "number"
    else:
        # All other entities use id as display
        return "id"

# ── merge triple ─────────────────────────────────────────────────
def merge(tx, t):
    sl  = label_from(t.get("sub_type"))
    ol  = label_from(t.get("obj_type"))
    sk  = key(sl);  dk = disp(sl)
    ok  = key(ol);  dk2 = disp(ol)

    # Ensure subject and object are strings before canonicalization
    sub_val = str(t["sub"]) if not isinstance(t["sub"], str) else t["sub"]
    obj_val = str(t["obj"]) if not isinstance(t["obj"], str) else t["obj"]

    sc  = canon(sub_val)
    oc  = canon(obj_val)

    # Convert decimal values to float for Neo4j compatibility
    props = {k: float(v) if isinstance(v, decimal.Decimal) else v
             for k, v in t.get("props", {}).items()}

    # Create a hash for deduplication
    h = hashlib.md5(json.dumps(t, sort_keys=True, default=str).encode()).hexdigest()

    # Run Cypher query to merge the triple
    tx.run(f"""
    MERGE (s:{sl} {{ {sk}: $sc }})
      ON CREATE SET s.{dk} = $sdisp
    SET  s.case_id = coalesce(s.case_id,$cid)
    MERGE (o:{ol} {{ {ok}: $oc }})
      ON CREATE SET o.{dk2} = $odisp
    SET  o.case_id = coalesce(o.case_id,$cid)
    MERGE (s)-[r:`{t['pred']}`]->(o)
    FOREACH (_ IN CASE WHEN r._hash IS NULL THEN [1] END |
        SET r += $props, r._hash = $h)
    """, sc=sc, oc=oc, sdisp=sub_val, odisp=obj_val,
         props=props, h=h, cid=t.get("case_id"))

# ── merge person relation ─────────────────────────────────────────
def merge_person_relation(tx, person_name, predicate, obj_val, obj_type, props, case_id=None):
    """Helper function to create common person-related triples"""
    triple = {
        "case_id": case_id,
        "sub": person_name,
        "sub_type": "PERSON",
        "pred": predicate,
        "obj": obj_val,
        "obj_type": obj_type,
        "props": props
    }
    merge(tx, triple)

# ── process documents ────────────────────────────────────────────
def process_documents(conn):
    """Process documents in batches using regular cursor instead of named cursor"""
    with conn.cursor() as cursor:
        # Get unprocessed document IDs first (to avoid cursor timeout issues)
        cursor.execute("""
            SELECT chunk_id FROM documents
            WHERE chunk_id NOT IN (SELECT source_id FROM kg_loaded)
        """)
        chunk_ids = [row[0] for row in cursor.fetchall()]

    # Process each document ID individually
    for chunk_id in chunk_ids:
        with conn.cursor() as cursor:
            # Fetch the specific document
            cursor.execute("""
                SELECT extra FROM documents WHERE chunk_id = %s
            """, (chunk_id,))

            row = cursor.fetchone()
            if not row:
                continue

            extra_json = row[0]

            # Parse the extra JSON if needed
            if isinstance(extra_json, str):
                try:
                    extra = json.loads(extra_json or '{}')
                except json.JSONDecodeError:
                    extra = {}
            else:
                extra = extra_json or {}

            # Skip if no triples
            triples = extra.get("triples", [])
            if not triples:
                # Mark as processed even if no triples
                with conn.cursor() as c2:
                    c2.execute("INSERT INTO kg_loaded(source_id) VALUES (%s)", (chunk_id,))
                    conn.commit()
                continue

            # Process triples for this document
            process_triples(chunk_id, triples)

            # Mark as processed
            with conn.cursor() as c2:
                c2.execute("INSERT INTO kg_loaded(source_id) VALUES (%s)", (chunk_id,))
                conn.commit()

# ── process triples ────────────────────────────────────────────────
def process_triples(chunk_id, triples):
    """Process all triples for a document"""
    # Create a new Neo4j session for each document
    with NEO.session() as neo_doc:
        # Process each triple individually
        for t in triples:
            try:
                # Use a separate transaction for each triple
                with neo_doc.begin_transaction() as tx:
                    merge(tx, t)
                    tx.commit()
            except exceptions.ConstraintError as ce:
                print(f"Constraint error for triple in chunk {chunk_id}: {ce}")
                continue
            except Exception as e:
                print(f"Error processing triple in chunk {chunk_id}: {e}")
                continue

# ── DAG callable ────────────────────────────────────────────────
def build_graph():
    # Set up Neo4j constraints first in its own session
    with NEO.session() as neo_setup:
        for c in CONSTRAINTS:
            try:
                neo_setup.run(c)
            except Exception as e:
                print(f"Error creating constraint: {e}")
                continue

    # Connect to PostgreSQL and process documents
    try:
        with psycopg2.connect(**PG) as pg:
            process_documents(pg)
    except Exception as e:
        print(f"Database error: {e}")
        raise

# ── Airflow DAG wrapper ──────────────────────────────────────────
with DAG("kg_builder",
         start_date=datetime(2024,1,1),
         schedule_interval=None,
         catchup=False) as dag:
    PythonOperator(task_id="build_graph", python_callable=build_graph)