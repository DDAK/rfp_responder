"""
vector_loader.py  –  semantic‑chunk Groq loader
================================================

Writes one row per ~150‑token chunk to Postgres.documents
and fills `extra` with triples ready for kg_builder.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import fitz, psycopg2, json, re, uuid, os, io, tempfile
from decimal import Decimal
from pathlib import Path
from langdetect import detect
import groq, tiktoken, requests, decimal
from minio import Minio

# ─── Groq client and models ───────────────────────────────────────
groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY", ""))
EMBED_MODEL = "text-embedding-3-small"        # 1 536‑dim
CHAT_MODEL  = "llama-3.3-70b-versatile"

enc = tiktoken.encoding_for_model("gpt-4")    # any cl100k model works

# ─── MinIO client setup ─────────────────────────────────────────────────────
minio_client = Minio(
    endpoint=os.getenv("MINIO_ENDPOINT", "minio:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    secure=False  # Set to True if using HTTPS
)
BUCKET_NAME = "data-bucket"

# ─── helper: Groq embeddings -------------------------------------------------
def embed(text, lang='en'):
    LANG_MODELS = {"en": "llama3:8b", "ar": "llama3:8b"}
    EMBEDDING_URL = "http://ollama:11434/api/embeddings"

    r = requests.post(EMBEDDING_URL,
        json={"model": LANG_MODELS[lang], "prompt": text[:2000]},
        timeout=60
    ).json()
    if "embedding" not in r:
        raise RuntimeError(f"Ollama error: {r}")
    return r["embedding"][:1536]          # truncate

# ─── helper: JSON extraction -------------------------------------------------
def llm_extract(prompt: str, text: str, cap: int = 4000) -> dict:
    chat = groq_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":f"{prompt}\n```{text[:cap]}```"}],
        stream=False
    )
    content = chat.choices[0].message.content
    block = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
    json_str = block.group(1) if block else re.search(r'{[\s\S]*}', content).group(0)
    print(json_str)
    return json.loads(json_str)

# ─── helper: split into ~150‑token chunks ------------------------------------
def semantic_chunks(text: str, max_tokens: int = 150):
    words = text.split()
    chunk, chunk_tokens = [], 0
    for w in words:
        tks = len(enc.encode(w + " "))
        if chunk_tokens + tks > max_tokens:
            yield " ".join(chunk)
            chunk, chunk_tokens = [], 0
        chunk.append(w)
        chunk_tokens += tks
    if chunk:
        yield " ".join(chunk)

# ─── document classifier -----------------------------------------------------
def classify(text: str):
    up = text.upper()
    if "CHILD SUPPORT GUIDELINES"     in up: return "policy"
    if "CHILD SUPPORT WORKSHEET"      in up: return "worksheet"
    if "DOMESTIC RELATIONS AFFIDAVIT" in up: return "affidavit"
    if "SALARY SLIP"                  in up: return "salary_slip"
    return None

# ─── Decimal → float for JSON -----------------------------------------------
def clean(obj):
    if isinstance(obj, Decimal): return float(obj)
    if isinstance(obj, dict):    return {k: clean(v) for k,v in obj.items()}
    if isinstance(obj, list):    return [clean(v) for v in obj]
    return obj


def safe_decimal(value, default="0"):
    """Safely convert a value to Decimal, handling None, empty strings, and formatting issues"""
    if value is None or value == "":
        return decimal.Decimal(default)

    # If it's already a number type, convert directly
    if isinstance(value, (int, float, decimal.Decimal)):
        return decimal.Decimal(str(value))

    # For strings, try to clean up common formatting issues
    try:
        # Remove currency symbols, commas, etc.
        clean_value = str(value).replace("$", "").replace(",", "").strip()
        if clean_value == "":
            return decimal.Decimal(default)
        return decimal.Decimal(clean_value)
    except (decimal.InvalidOperation, ValueError):
        print(f"Warning: Could not convert '{value}' to Decimal, using default {default}")
        return decimal.Decimal(default)

# ─── Helper: list files from MinIO bucket ───────────────────────────────────
def list_minio_files(prefix):
    objects = minio_client.list_objects(BUCKET_NAME, prefix=prefix, recursive=True)
    return [obj.object_name for obj in objects]

# ─── Helper: download file from MinIO to temp file ---------------------------
def download_from_minio(object_name):
    temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    try:
        # Get object data
        response = minio_client.get_object(BUCKET_NAME, object_name)
        # Write to temp file
        for d in response.stream(32*1024):
            temp_file.write(d)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise Exception(f"Error downloading {object_name}: {str(e)}")

# ─── Helper: Mark file as processed by copying to a separate file -----------
def mark_as_processed(object_name):
    """Mark a file as processed by creating a marker file in the processed directory"""
    dir_path = os.path.dirname(object_name)
    file_name = os.path.basename(object_name)
    processed_marker = f"{dir_path}/processed/{file_name}.processed"

    # Create an empty marker file
    minio_client.put_object(
        BUCKET_NAME,
        processed_marker,
        io.BytesIO(b''),
        0  # Zero length
    )
    print(f"Marked {object_name} as processed with marker: {processed_marker}")

# ─── Helper: Ensure processed directories exist -----------------------------
def ensure_processed_dirs():
    # List of subdirectories that need processed folders
    subdirs = ['raw/affidavit', 'raw/worksheets', 'raw/policy', 'raw/salaryslips']

    for subdir in subdirs:
        processed_dir = f"{subdir}/processed/"
        try:
            # Try to list objects in the processed directory
            objects = list(minio_client.list_objects(BUCKET_NAME, prefix=processed_dir, recursive=False))
            if len(objects) == 0:
                # Directory doesn't exist or is empty, create an empty object to represent it
                minio_client.put_object(BUCKET_NAME, processed_dir, io.BytesIO(b''), 0)
                print(f"Created directory: {processed_dir}")
        except Exception as e:
            print(f"Error creating directory {processed_dir}: {e}")
            # Still try to create it
            try:
                minio_client.put_object(BUCKET_NAME, processed_dir, io.BytesIO(b''), 0)
            except:
                pass

# ─── Helper: Check if a file is already processed ---------------------------
def is_processed(object_name):
    """Check if there's a marker file indicating this file was processed"""
    dir_path = os.path.dirname(object_name)
    file_name = os.path.basename(object_name)
    processed_marker = f"{dir_path}/processed/{file_name}.processed"

    try:
        minio_client.stat_object(BUCKET_NAME, processed_marker)
        return True
    except:
        return False

# ─── main loader callable ----------------------------------------------------
def load_vectors():
    pg = psycopg2.connect(host="postgres", dbname="demo",
                          user="demo", password="demo1234")

    # Ensure processed directories exist
    ensure_processed_dirs()

    with pg, pg.cursor() as cur:
        # List PDF files in MinIO bucket
        pdf_files = []
        for subfolder in ['affidavit', 'worksheets', 'policy', 'salaryslips']:
            files = list_minio_files(f"raw/{subfolder}")
            # Filter out the "processed" directory and its contents
            files = [f for f in files if "/processed/" not in f and f.endswith('.pdf')]
            pdf_files.extend(files)

        for object_name in pdf_files:
            # Skip if already processed
            if is_processed(object_name):
                print(f"Skipping already processed file: {object_name}")
                continue

            pdf_name = Path(object_name).name
            print(f"Processing: {object_name}")

            # Download file to temporary location
            try:
                temp_path = download_from_minio(object_name)

                # Process the PDF
                doc = fitz.open(temp_path)
                first = doc.load_page(0).get_text("text")
                dtype = classify(first)
                if not dtype:
                    print(f"Skip {object_name} - unable to classify")
                    # Mark as processed even if we can't classify it
                    mark_as_processed(object_name)
                    os.unlink(temp_path)
                    continue

                meta = {"doc_type": dtype, "triples": []}
                full = "".join(p.get_text("text") for p in doc)

                # ---- Policy (LLM) ----------------------------------------
                if dtype == "policy":
                    meta.update(llm_extract(
                        "Return JSON {issuer,effective_date,sections:[{id,title}]}",
                        full))
                    chunks = list(semantic_chunks(full))

                # ---- Worksheet (LLM) ------------------------------------
                elif dtype == "worksheet":
                    # Extract all needed data in one call
                    data = llm_extract(
                        "Return JSON {case_id,father,mother,"
                        "gross_income_father,gross_income_mother,"
                        "net_obligation_father,net_obligation_mother,"
                        "combined_income,children_age_0_5,children_age_6_11,children_age_12_18,"
                        "gross_child_support,father_proportionate_share,mother_proportionate_share,"
                        "father_income_proportion,mother_income_proportion,"
                        "father_parenting_adjustment,mother_parenting_adjustment,"
                        "health_insurance_premium,father_health_insurance,mother_health_insurance,"
                        "child_care_costs,father_child_care_share,mother_child_care_share,"
                        "calculation_date}", full)

                    meta.update(data)

                    # Convert to appropriate data types
                    fi = Decimal(str(data["gross_income_father"]))
                    mi = Decimal(str(data["gross_income_mother"]))
                    fnet = Decimal(str(data["net_obligation_father"]))
                    mnet = Decimal(str(data["net_obligation_mother"]))
                    combined_income = Decimal(str(data.get("combined_income", "0")))
                    gross_support = Decimal(str(data.get("gross_child_support", "0")))
                    father_prop_share = Decimal(str(data.get("father_proportionate_share", "0")))
                    mother_prop_share = Decimal(str(data.get("mother_proportionate_share", "0")))
                    father_parent_adj = Decimal(str(data.get("father_parenting_adjustment", "0")))
                    mother_parent_adj = Decimal(str(data.get("mother_parenting_adjustment", "0")))
                    health_premium = Decimal(str(data.get("health_insurance_premium", "0")))
                    father_health = Decimal(str(data.get("father_health_insurance", "0")))
                    mother_health = Decimal(str(data.get("mother_health_insurance", "0")))
                    child_care = Decimal(str(data.get("child_care_costs", "0")))
                    father_child_care = Decimal(str(data.get("father_child_care_share", "0")))
                    mother_child_care = Decimal(str(data.get("mother_child_care_share", "0")))

                    # Get other data types
                    father_percent = data.get("father_income_proportion", "0%")
                    mother_percent = data.get("mother_income_proportion", "0%")
                    children_0_5 = int(data.get("children_age_0_5", 0))
                    children_6_11 = int(data.get("children_age_6_11", 0))
                    children_12_18 = int(data.get("children_age_12_18", 0))
                    total_children = children_0_5 + children_6_11 + children_12_18

                    meta["triples"] += [
                        # Original triples
                        {"case_id": data["case_id"],
                         "sub": data["father"], "sub_type": "PERSON",
                         "pred": "HAS_GROSS_INCOME",
                         "obj": f"income:{fi}", "obj_type": "INCOME",
                         "props": {"amount": fi, "period": "monthly"}},

                        {"sub": data["mother"], "sub_type": "PERSON",
                         "pred": "HAS_GROSS_INCOME",
                         "obj": f"income:{mi}", "obj_type": "INCOME",
                         "props": {"amount": mi, "period": "monthly"}},

                        {"sub": data["father"], "pred": "HAS_NET_INCOME",
                         "obj": f"income:{fnet}", "obj_type": "INCOME",
                         "props": {"amount": fnet, "period": "monthly"}},

                        {"sub": data["mother"], "pred": "HAS_NET_INCOME",
                         "obj": f"income:{mnet}", "obj_type": "INCOME",
                         "props": {"amount": mnet, "period": "monthly"}},

                        # Additional case relationships
                        {"case_id": data["case_id"],
                         "sub": f"case:{data['case_id']}", "sub_type": "CASE",
                         "pred": "HAS_DOCUMENT_TYPE",
                         "obj": "Child Support Worksheet", "obj_type": "DOCUMENT_TYPE",
                         "props": {}},

                        {"case_id": data["case_id"],
                         "sub": data["father"], "sub_type": "PERSON",
                         "pred": "IS_PARENT_IN",
                         "obj": f"case:{data['case_id']}", "obj_type": "CASE",
                         "props": {"role": "parent"}},

                        {"case_id": data["case_id"],
                         "sub": data["mother"], "sub_type": "PERSON",
                         "pred": "IS_PARENT_IN",
                         "obj": f"case:{data['case_id']}", "obj_type": "CASE",
                         "props": {"role": "parent"}},

                        {"case_id": data["case_id"],
                         "sub": f"case:{data['case_id']}", "sub_type": "CASE",
                         "pred": "HAS_COMBINED_INCOME",
                         "obj": f"income:{combined_income}", "obj_type": "INCOME",
                         "props": {"amount": combined_income, "currency": "USD"}},

                        {"case_id": data["case_id"],
                         "sub": f"case:{data['case_id']}", "sub_type": "CASE",
                         "pred": "HAS_CHILDREN",
                         "obj": "children", "obj_type": "CHILDREN",
                         "props": {"age_0_5_count": children_0_5,
                                   "age_6_11_count": children_6_11,
                                   "age_12_18_count": children_12_18,
                                   "total_count": total_children}},

                        {"case_id": data["case_id"],
                         "sub": f"case:{data['case_id']}", "sub_type": "CASE",
                         "pred": "HAS_GROSS_CHILD_SUPPORT_OBLIGATION",
                         "obj": f"obligation:{gross_support}", "obj_type": "OBLIGATION",
                         "props": {"amount": gross_support, "currency": "USD"}},

                        {"case_id": data["case_id"],
                         "sub": f"case:{data['case_id']}", "sub_type": "CASE",
                         "pred": "HAS_HEALTH_INSURANCE_PREMIUM",
                         "obj": f"premium:{health_premium}", "obj_type": "PREMIUM",
                         "props": {"amount": health_premium, "currency": "USD"}},

                        {"case_id": data["case_id"],
                         "sub": f"case:{data['case_id']}", "sub_type": "CASE",
                         "pred": "HAS_CHILD_CARE_COSTS",
                         "obj": f"costs:{child_care}", "obj_type": "COSTS",
                         "props": {"type": "work_related", "amount": child_care, "currency": "USD"}},

                        {"case_id": data["case_id"],
                         "sub": f"case:{data['case_id']}", "sub_type": "CASE",
                         "pred": "WAS_CALCULATED_ON",
                         "obj": data.get("calculation_date", ""), "obj_type": "DATE",
                         "props": {}},

                        # Additional father relationships
                        {"case_id": data["case_id"],
                         "sub": data["father"], "sub_type": "PERSON",
                         "pred": "HAS_INCOME_PROPORTION",
                         "obj": f"proportion:{father_percent}", "obj_type": "PROPORTION",
                         "props": {"percentage": father_percent, "calculation_basis": "combined_income"}},

                        {"case_id": data["case_id"],
                         "sub": data["father"], "sub_type": "PERSON",
                         "pred": "HAS_PROPORTIONATE_SHARE",
                         "obj": f"share:{father_prop_share}", "obj_type": "SHARE",
                         "props": {"amount": father_prop_share, "share_type": "child_support", "currency": "USD"}},

                        {"case_id": data["case_id"],
                         "sub": data["father"], "sub_type": "PERSON",
                         "pred": "HAS_PARENTING_TIME_ADJUSTMENT",
                         "obj": f"adjustment:{father_parent_adj}", "obj_type": "ADJUSTMENT",
                         "props": {"amount": father_parent_adj, "adjustment_type": "20%", "currency": "USD"}},

                        {"case_id": data["case_id"],
                         "sub": data["father"], "sub_type": "PERSON",
                         "pred": "PAYS_HEALTH_INSURANCE",
                         "obj": f"payment:{father_health}", "obj_type": "PAYMENT",
                         "props": {"amount": father_health, "type": "health_and_dental", "currency": "USD"}},

                        {"case_id": data["case_id"],
                         "sub": data["father"], "sub_type": "PERSON",
                         "pred": "HAS_PROPORTIONATE_SHARE_CHILD_CARE",
                         "obj": f"share:{father_child_care}", "obj_type": "SHARE",
                         "props": {"amount": father_child_care, "currency": "USD"}},

                        {"case_id": data["case_id"],
                         "sub": data["father"], "sub_type": "PERSON",
                         "pred": "HAS_CHILD_SUPPORT_OBLIGATION",
                         "obj": f"obligation:{fnet}", "obj_type": "OBLIGATION",
                         "props": {"amount": fnet, "obligation_type": "basic", "currency": "USD"}},

                        # Additional mother relationships
                        {"case_id": data["case_id"],
                         "sub": data["mother"], "sub_type": "PERSON",
                         "pred": "HAS_INCOME_PROPORTION",
                         "obj": f"proportion:{mother_percent}", "obj_type": "PROPORTION",
                         "props": {"percentage": mother_percent, "calculation_basis": "combined_income"}},

                        {"case_id": data["case_id"],
                         "sub": data["mother"], "sub_type": "PERSON",
                         "pred": "HAS_PROPORTIONATE_SHARE",
                         "obj": f"share:{mother_prop_share}", "obj_type": "SHARE",
                         "props": {"amount": mother_prop_share, "share_type": "child_support", "currency": "USD"}},

                        {"case_id": data["case_id"],
                         "sub": data["mother"], "sub_type": "PERSON",
                         "pred": "HAS_PARENTING_TIME_ADJUSTMENT",
                         "obj": f"adjustment:{mother_parent_adj}", "obj_type": "ADJUSTMENT",
                         "props": {"amount": mother_parent_adj, "adjustment_type": "20%", "currency": "USD"}},

                        {"case_id": data["case_id"],
                         "sub": data["mother"], "sub_type": "PERSON",
                         "pred": "PAYS_HEALTH_INSURANCE",
                         "obj": f"payment:{mother_health}", "obj_type": "PAYMENT",
                         "props": {"amount": mother_health, "type": "health_and_dental", "currency": "USD"}},

                        {"case_id": data["case_id"],
                         "sub": data["mother"], "sub_type": "PERSON",
                         "pred": "HAS_PROPORTIONATE_SHARE_CHILD_CARE",
                         "obj": f"share:{mother_child_care}", "obj_type": "SHARE",
                         "props": {"amount": mother_child_care, "currency": "USD"}},

                        {"case_id": data["case_id"],
                         "sub": data["mother"], "sub_type": "PERSON",
                         "pred": "HAS_CHILD_SUPPORT_OBLIGATION",
                         "obj": f"obligation:{mnet}", "obj_type": "OBLIGATION",
                         "props": {"amount": mnet, "obligation_type": "basic", "currency": "USD"}},
                    ]
                    chunks = list(semantic_chunks(full))
                # ---- Affidavit (LLM) ------------------------------------
                elif dtype == "affidavit":
                    # Extract basic case information and detailed entities
                    data = llm_extract(
                        "Return JSON {case_id,court,county,state,"
                        "petitioner,respondent,petitioner_address,respondent_address,"
                        "petitioner_ssn,respondent_ssn,petitioner_birth,respondent_birth,"
                        "petitioner_phone,respondent_phone,marriage_date,petitioner_marriage_count,"
                        "respondent_marriage_count,child_count,children,petitioner_employer,"
                        "respondent_employer,petitioner_gross_income,respondent_gross_income,"
                        "petitioner_net_income,respondent_net_income,petitioner_assets,respondent_assets}", full)

                    meta.update(data)
                    case_id = data["case_id"]

                    # Base triples for case information
                    meta["triples"] += [
                        # Case information
                        {"case_id": case_id,
                         "sub": f"case:{case_id}", "sub_type": "CASE",
                         "pred": "HAS_TYPE", "obj": "Domestic Relations", "obj_type": "CASE_TYPE",
                         "props": {"court": data.get("court", ""),
                                   "county": data.get("county", ""),
                                   "state": data.get("state", "")}},

                        # Petitioner relationships
                        {"case_id": case_id,
                         "sub": data["petitioner"], "sub_type": "PERSON",
                         "pred": "IS_PARTY_IN", "obj": f"case:{case_id}", "obj_type": "CASE",
                         "props": {"role": "petitioner"}},

                        {"case_id": case_id,
                         "sub": data["petitioner"], "sub_type": "PERSON",
                         "pred": "HAS_ADDRESS", "obj": data.get("petitioner_address", ""), "obj_type": "ADDRESS",
                         "props": {"type": "residence"}},

                        # Respondent relationships
                        {"case_id": case_id,
                         "sub": data["respondent"], "sub_type": "PERSON",
                         "pred": "IS_PARTY_IN", "obj": f"case:{case_id}", "obj_type": "CASE",
                         "props": {"role": "respondent"}},

                        {"case_id": case_id,
                         "sub": data["respondent"], "sub_type": "PERSON",
                         "pred": "HAS_ADDRESS", "obj": data.get("respondent_address", ""), "obj_type": "ADDRESS",
                         "props": {"type": "residence"}},

                        # Marriage relationship
                        {"case_id": case_id,
                         "sub": data["petitioner"], "sub_type": "PERSON",
                         "pred": "MARRIED_TO", "obj": data["respondent"], "obj_type": "PERSON",
                         "props": {"date": data.get("marriage_date", "")}}
                    ]

                    # Add SSN information if available
                    if "petitioner_ssn" in data and data["petitioner_ssn"]:
                        meta["triples"].append({
                            "case_id": case_id,
                            "sub": data["petitioner"], "sub_type": "PERSON",
                            "pred": "HAS_SSN", "obj": data["petitioner_ssn"], "obj_type": "SSN",
                            "props": {"partial": True if "X" in data["petitioner_ssn"] else False}
                        })

                    if "respondent_ssn" in data and data["respondent_ssn"]:
                        meta["triples"].append({
                            "case_id": case_id,
                            "sub": data["respondent"], "sub_type": "PERSON",
                            "pred": "HAS_SSN", "obj": data["respondent_ssn"], "obj_type": "SSN",
                            "props": {"partial": True if "X" in data["respondent_ssn"] else False}
                        })

                    # Add birth date information if available
                    if "petitioner_birth" in data and data["petitioner_birth"]:
                        meta["triples"].append({
                            "case_id": case_id,
                            "sub": data["petitioner"], "sub_type": "PERSON",
                            "pred": "HAS_BIRTH_DATE", "obj": data["petitioner_birth"], "obj_type": "DATE",
                            "props": {"format": "month/year" if "/" in data["petitioner_birth"] else ""}
                        })

                    if "respondent_birth" in data and data["respondent_birth"]:
                        meta["triples"].append({
                            "case_id": case_id,
                            "sub": data["respondent"], "sub_type": "PERSON",
                            "pred": "HAS_BIRTH_DATE", "obj": data["respondent_birth"], "obj_type": "DATE",
                            "props": {"format": "month/year" if "/" in data["respondent_birth"] else ""}
                        })

                    # Add phone information if available
                    if "petitioner_phone" in data and data["petitioner_phone"]:
                        meta["triples"].append({
                            "case_id": case_id,
                            "sub": data["petitioner"], "sub_type": "PERSON",
                            "pred": "HAS_PHONE", "obj": data["petitioner_phone"], "obj_type": "PHONE",
                            "props": {"type": "telephone"}
                        })

                    if "respondent_phone" in data and data["respondent_phone"]:
                        meta["triples"].append({
                            "case_id": case_id,
                            "sub": data["respondent"], "sub_type": "PERSON",
                            "pred": "HAS_PHONE", "obj": data["respondent_phone"], "obj_type": "PHONE",
                            "props": {"type": "telephone"}
                        })

                    # Add marriage count information if available
                    if "petitioner_marriage_count" in data and data["petitioner_marriage_count"]:
                        meta["triples"].append({
                            "case_id": case_id,
                            "sub": data["petitioner"], "sub_type": "PERSON",
                            "pred": "HAS_MARRIAGE_COUNT", "obj": data["petitioner_marriage_count"], "obj_type": "COUNT",
                            "props": {"as_of_document_date": True}
                        })

                    if "respondent_marriage_count" in data and data["respondent_marriage_count"]:
                        meta["triples"].append({
                            "case_id": case_id,
                            "sub": data["respondent"], "sub_type": "PERSON",
                            "pred": "HAS_MARRIAGE_COUNT", "obj": data["respondent_marriage_count"], "obj_type": "COUNT",
                            "props": {"as_of_document_date": True}
                        })

                    # Add employment information if available
                    if "petitioner_employer" in data and data["petitioner_employer"]:
                        meta["triples"].append({
                            "case_id": case_id,
                            "sub": data["petitioner"], "sub_type": "PERSON",
                            "pred": "IS_EMPLOYED_BY", "obj": data["petitioner_employer"], "obj_type": "ORG",
                            "props": {}
                        })

                    if "respondent_employer" in data and data["respondent_employer"]:
                        meta["triples"].append({
                            "case_id": case_id,
                            "sub": data["respondent"], "sub_type": "PERSON",
                            "pred": "IS_EMPLOYED_BY", "obj": data["respondent_employer"], "obj_type": "ORG",
                            "props": {}
                        })

                    # Add income information if available
                    if "petitioner_gross_income" in data and data["petitioner_gross_income"]:
                        meta["triples"].append({
                            "case_id": case_id,
                            "sub": data["petitioner"], "sub_type": "PERSON",
                            "pred": "HAS_GROSS_INCOME",
                            "obj": f"income:{data['petitioner_gross_income']}", "obj_type": "INCOME",
                            "props": {"amount": data["petitioner_gross_income"], "currency": "USD",
                                      "frequency": "monthly"}
                        })

                    if "respondent_gross_income" in data and data["respondent_gross_income"]:
                        meta["triples"].append({
                            "case_id": case_id,
                            "sub": data["respondent"], "sub_type": "PERSON",
                            "pred": "HAS_GROSS_INCOME",
                            "obj": f"income:{data['respondent_gross_income']}", "obj_type": "INCOME",
                            "props": {"amount": data["respondent_gross_income"], "currency": "USD",
                                      "frequency": "monthly"}
                        })

                    if "petitioner_net_income" in data and data["petitioner_net_income"]:
                        meta["triples"].append({
                            "case_id": case_id,
                            "sub": data["petitioner"], "sub_type": "PERSON",
                            "pred": "HAS_NET_INCOME",
                            "obj": f"income:{data['petitioner_net_income']}", "obj_type": "INCOME",
                            "props": {"amount": data["petitioner_net_income"], "currency": "USD",
                                      "frequency": "monthly"}
                        })

                    if "respondent_net_income" in data and data["respondent_net_income"]:
                        meta["triples"].append({
                            "case_id": case_id,
                            "sub": data["respondent"], "sub_type": "PERSON",
                            "pred": "HAS_NET_INCOME",
                            "obj": f"income:{data['respondent_net_income']}", "obj_type": "INCOME",
                            "props": {"amount": data["respondent_net_income"], "currency": "USD",
                                      "frequency": "monthly"}
                        })

                    # Process children information if available
                    if "children" in data and isinstance(data["children"], list):
                        for child in data["children"]:
                            if isinstance(child, dict):
                                child_name = child.get("name", "")
                                if child_name:
                                    # Child relationship to petitioner
                                    meta["triples"].append({
                                        "case_id": case_id,
                                        "sub": child_name, "sub_type": "PERSON",
                                        "pred": "IS_CHILD_OF", "obj": data["petitioner"], "obj_type": "PERSON",
                                        "props": {"relationship": "biological"}
                                    })

                                    # Child relationship to respondent
                                    meta["triples"].append({
                                        "case_id": case_id,
                                        "sub": child_name, "sub_type": "PERSON",
                                        "pred": "IS_CHILD_OF", "obj": data["respondent"], "obj_type": "PERSON",
                                        "props": {"relationship": "biological"}
                                    })

                                    # Child SSN if available
                                    if "ssn" in child and child["ssn"]:
                                        meta["triples"].append({
                                            "case_id": case_id,
                                            "sub": child_name, "sub_type": "PERSON",
                                            "pred": "HAS_SSN", "obj": child["ssn"], "obj_type": "SSN",
                                            "props": {"partial": True if "X" in child["ssn"] else False}
                                        })

                                    # Child birth date if available
                                    if "birth_date" in child and child["birth_date"]:
                                        meta["triples"].append({
                                            "case_id": case_id,
                                            "sub": child_name, "sub_type": "PERSON",
                                            "pred": "HAS_BIRTH_DATE", "obj": child["birth_date"], "obj_type": "DATE",
                                            "props": {"format": "month/year" if "/" in child["birth_date"] else ""}
                                        })

                                    # Child age if available
                                    if "age" in child and child["age"]:
                                        meta["triples"].append({
                                            "case_id": case_id,
                                            "sub": child_name, "sub_type": "PERSON",
                                            "pred": "HAS_AGE", "obj": child["age"], "obj_type": "AGE",
                                            "props": {"as_of_document_date": True}
                                        })

                                    # Child custody if available
                                    if "custody" in child and child["custody"]:
                                        meta["triples"].append({
                                            "case_id": case_id,
                                            "sub": child_name, "sub_type": "PERSON",
                                            "pred": "HAS_CUSTODY_ARRANGEMENT", "obj": child["custody"],
                                            "obj_type": "CUSTODY",
                                            "props": {"parties": [data["petitioner"], data["respondent"]]}
                                        })

                    # Process assets information if available
                    if "petitioner_assets" in data and isinstance(data["petitioner_assets"], list):
                        for asset in data["petitioner_assets"]:
                            if isinstance(asset, dict):
                                asset_type = asset.get("type", "")
                                asset_name = asset.get("name", "")
                                asset_value = asset.get("value", "")

                                if asset_type and asset_name:
                                    meta["triples"].append({
                                        "case_id": case_id,
                                        "sub": data["petitioner"], "sub_type": "PERSON",
                                        "pred": f"HAS_{asset_type.upper()}", "obj": asset_name,
                                        "obj_type": asset_type.upper(),
                                        "props": {"balance": asset_value, "currency": "USD",
                                                  "ownership": asset.get("ownership", "Individual")}
                                    })

                    if "respondent_assets" in data and isinstance(data["respondent_assets"], list):
                        for asset in data["respondent_assets"]:
                            if isinstance(asset, dict):
                                asset_type = asset.get("type", "")
                                asset_name = asset.get("name", "")
                                asset_value = asset.get("value", "")

                                if asset_type and asset_name:
                                    meta["triples"].append({
                                        "case_id": case_id,
                                        "sub": data["respondent"], "sub_type": "PERSON",
                                        "pred": f"HAS_{asset_type.upper()}", "obj": asset_name,
                                        "obj_type": asset_type.upper(),
                                        "props": {"balance": asset_value, "currency": "USD",
                                                  "ownership": asset.get("ownership", "Individual")}
                                    })

                    chunks = list(semantic_chunks(full))

                # ---- Salary slip (LLM) ----------------------------------
                elif dtype == "salary_slip":
                    data = llm_extract(
                        "Return JSON {employee,employee_id,department,designation,"
                        "employer,employer_address,employer_phone,employer_email,"
                        "pay_period_start,pay_period_end,pay_date,"
                        "bank_name,account_number,payment_method,payment_id,"
                        "basic_salary,performance_bonus,overtime_rate,overtime_hours,other_allowances,gross,"
                        "federal_tax,social_security,medicare,state_tax,health_insurance,total_deductions,net,"
                        "ytd_gross,ytd_deductions,ytd_net,ytd_taxable,ytd_401k,leave_balance}", full)

                    meta.update(data)

                    # Convert numerical values to Decimal
                    g = safe_decimal(data.get("gross"))
                    n = safe_decimal(data.get("net"))
                    basic = safe_decimal(data.get("basic_salary"))
                    bonus = safe_decimal(data.get("performance_bonus"))
                    fed_tax = safe_decimal(data.get("federal_tax"))
                    ss_tax = safe_decimal(data.get("social_security"))
                    medicare_tax = safe_decimal(data.get("medicare"))
                    state_tax = safe_decimal(data.get("state_tax"))
                    health_ins = safe_decimal(data.get("health_insurance"))
                    total_deduct = safe_decimal(data.get("total_deductions"))

                    # YTD values
                    ytd_gross = safe_decimal(data.get("ytd_gross"))
                    ytd_net = safe_decimal(data.get("ytd_net"))
                    ytd_deduct = safe_decimal(data.get("ytd_deductions"))
                    ytd_taxable = safe_decimal(data.get("ytd_taxable"))
                    ytd_401k = safe_decimal(data.get("ytd_401k"))

                    # Extract pay period
                    period_start = data.get("pay_period_start", "")
                    period_end = data.get("pay_period_end", "")
                    pay_date = data.get("pay_date", "")

                    # Generate document ID if no case ID
                    doc_id = data.get("case_id", f"payslip_{data['employee_id']}_{period_end.replace('-', '')}")

                    meta["triples"] += [
                        # Basic employment relationship
                        {"case_id": doc_id,
                         "sub": data["employee"], "sub_type": "PERSON",
                         "pred": "EMPLOYED_BY",
                         "obj": data["employer"], "obj_type": "ORG",
                         "props": {"department": data.get("department", ""),
                                   "designation": data.get("designation", ""),
                                   "employee_id": data.get("employee_id", "")}},

                        # Organizational information
                        {"case_id": doc_id,
                         "sub": data["employer"], "sub_type": "ORG",
                         "pred": "HAS_ADDRESS",
                         "obj": data.get("employer_address", ""), "obj_type": "ADDRESS",
                         "props": {"type": "business"}},

                        {"case_id": doc_id,
                         "sub": data["employer"], "sub_type": "ORG",
                         "pred": "HAS_CONTACT",
                         "obj": f"contact:{data.get('employer_phone', '')}", "obj_type": "CONTACT",
                         "props": {"phone": data.get("employer_phone", ""),
                                   "email": data.get("employer_email", "")}},

                        # Payment information
                        {"case_id": doc_id,
                         "sub": data["employee"], "sub_type": "PERSON",
                         "pred": "HAS_PAYMENT",
                         "obj": f"payment:{data.get('payment_id', '')}", "obj_type": "PAYMENT",
                         "props": {"payment_id": data.get("payment_id", ""),
                                   "payment_method": data.get("payment_method", ""),
                                   "bank_name": data.get("bank_name", ""),
                                   "account_number": data.get("account_number", ""),
                                   "pay_date": pay_date,
                                   "pay_period_start": period_start,
                                   "pay_period_end": period_end}},

                        # Income information
                        {"case_id": doc_id,
                         "sub": data["employee"], "sub_type": "PERSON",
                         "pred": "HAS_GROSS_INCOME",
                         "obj": f"income:{g}", "obj_type": "INCOME",
                         "props": {"amount": g, "period": "monthly",
                                   "basic_salary": basic,
                                   "performance_bonus": bonus,
                                   "date": pay_date}},

                        {"case_id": doc_id,
                         "sub": data["employee"], "sub_type": "PERSON",
                         "pred": "HAS_NET_INCOME",
                         "obj": f"income:{n}", "obj_type": "INCOME",
                         "props": {"amount": n, "period": "monthly", "date": pay_date}},

                        # Deductions
                        {"case_id": doc_id,
                         "sub": data["employee"], "sub_type": "PERSON",
                         "pred": "HAS_DEDUCTIONS",
                         "obj": f"deductions:{total_deduct}", "obj_type": "DEDUCTIONS",
                         "props": {"amount": total_deduct,
                                   "federal_tax": fed_tax,
                                   "social_security": ss_tax,
                                   "medicare": medicare_tax,
                                   "state_tax": state_tax,
                                   "health_insurance": health_ins,
                                   "date": pay_date}},

                        # Year-to-date information
                        {"case_id": doc_id,
                         "sub": data["employee"], "sub_type": "PERSON",
                         "pred": "HAS_YTD_INCOME",
                         "obj": f"ytd_income:{ytd_gross}", "obj_type": "YTD_INCOME",
                         "props": {"gross_amount": ytd_gross,
                                   "net_amount": ytd_net,
                                   "taxable_wages": ytd_taxable,
                                   "total_deductions": ytd_deduct,
                                   "401k_contributions": ytd_401k,
                                   "year": "2025"}},

                        # Leave balance
                        {"case_id": doc_id,
                         "sub": data["employee"], "sub_type": "PERSON",
                         "pred": "HAS_LEAVE_BALANCE",
                         "obj": f"leave:{data.get('leave_balance', '0')}", "obj_type": "LEAVE",
                         "props": {"days": data.get("leave_balance", "0"),
                                   "as_of_date": pay_date}}
                    ]
                    chunks = list(semantic_chunks(full))

                # ---- insert every chunk --------------------------------
                for c_idx, chunk in enumerate(chunks):
                    if not chunk.strip(): continue
                    vec = embed(chunk)
                    cur.execute("""
                       INSERT INTO documents(
                         chunk_id, pdf_name, page_no, chunk_no,
                         text, token_count, embedding,
                         doc_type, case_id, extra)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT DO NOTHING
                    """, (
                        str(uuid.uuid4()), pdf_name, 0, c_idx,
                        chunk, len(enc.encode(chunk)), vec,
                        meta["doc_type"], meta.get("case_id"),
                        json.dumps(clean(meta))
                    ))

                # Mark file as processed after successful processing
                mark_as_processed(object_name)

                # Clean up
                os.unlink(temp_path)

            except Exception as e:
                print(f"Error processing {object_name}: {e}")
                raise

# ─── Airflow DAG -------------------------------------------------------------
with DAG(
    dag_id="vector_loader",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    PythonOperator(task_id="load_vectors", python_callable=load_vectors)