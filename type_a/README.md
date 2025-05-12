## Bring up the infra structure
```ssh
docker compose up -d
```
## Install the requirements
```ssh
python -m venv venv
source venv/bin/activate
pip install -r requirements
```
## Ingest the documents into vector db, KV store, knowledge graph, when data is in ./data

```ssh
python ingest/lrag.py
```
## rag query the KV store, vector db, Knowledge graph [modes: naive, local, global, hybrid]
```ssh
python rag-simple/qrag.py
```
## Agentic Proposal writer based on the knowledge in the KV store, vector db, Knowledge graph [modes: naive, local, global, hybrid]
```ssh
python rfp_agent.py
```
