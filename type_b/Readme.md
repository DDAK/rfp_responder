```ssh
docker compose build -d up
```
automatically ingests the file from minio in the neo4j Knowledge graph,
vector store pg_vector, KV store mongo


Worker listen on the queue (postgres queue) for a job; when we have job the worker executes the job from queue. Each job is an execution of agentic workflow that generates the proposal from langraph agent orchestration. The FastApi allows us to add jobs to the db queue.