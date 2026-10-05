# Durable thunc tasks

This is the optional Temporal beta. The ordinary thunc API remains local and does
not depend on Temporal. Durable mode needs a Temporal service, this application's
worker, and persistent worker storage. No production infrastructure is deployed by
installing thunc.

## Start locally

From the repository root:

```sh
pip install -e '.[temporal,openai]'
```

Install the [official Temporal CLI](https://docs.temporal.io/cli), then start its
local development service with an explicit persistent database:

```sh
mkdir -p .temporal-dev
temporal server start-dev --db-filename .temporal-dev/server.sqlite
```

The dev server is for local development, not a production deployment. Keep it
bound to loopback. In two other terminals, with your provider credentials already
configured in the worker environment:

```sh
cd examples/temporal
python worker.py
```

```sh
cd examples/temporal
python client.py
```

Put source files in `examples/temporal/temporal-workspace/` before starting the
worker, or set `THUNC_WORKSPACE` to an **exclusively worker-owned** directory. This
example is read-only and uses OpenAI. `THUNC_TEMPORAL_STATE` selects its persistent
journal/artifact directory, which must be outside the workspace. Never point the
example at a workspace with another active writer.

A repeated `timeout-review-1` request attaches to the same result. Change the
request ID for a deliberate new run. Identical request IDs with different inputs,
return schemas or deadlines are rejected. Keep IDs to 200 characters or fewer.

## API and lifecycle

```python
from thunc.temporal import Registry, Runtime, Worker

# Worker: only explicitly registered code can execute.
registry = Registry(state_dir="/srv/thunc-state")
registry.agent_task("repo.review", review_task, version="1", workspace_id="repo")
# Typed @thunc.function definitions use registry.function(...).
# An agent's own tools (tools=) run at most once; name those that may run again after a crash:
# registry.agent_task(..., retry_safe_tools=["find_issue"])

# Client: submitting and waiting are separate operations.
runtime = await Runtime.connect("localhost:7233", task_queue="repo-v1")
handle = await runtime.start(
    "repo.review",
    version="1",
    workspace_id="repo",
    inputs={},
    returns=str,
    request_id="review-123",
    deadline_seconds=1800,
)
run = await handle.result()  # DurableRun[str]; run.value is validated
same = runtime.get(handle.id, returns=str)
print(await same.status())
await same.cancel()
```

The result includes `steps`, Activity retry counts, `files_changed`, `commands`,
`denied`, `notes`, and immutable artifact hashes. `DurableError.record` retains the
run ID and terminal failure status. Detailed failures remain in Temporal history;
provider exception messages are sanitized. Local `Run.session` is unchanged.

Run states are `queued`, `running`, `needs_attention`, `completed`, `failed`,
`cancelled`, and `timed_out`. A disconnected submitting client does not cancel
anything. Deadlines include time queued after admission, and survive
Continue-As-New. A cancellation is cooperative: a command can already have had
effects. The workspace stays reserved until uncertain effects are reconciled.

`returns=` supports thunc's existing JSON-compatible return types, including
dataclasses. Worker-side task schemas are authoritative. Inputs require named,
fixed parameters; bound methods, positional-only parameters and variadic inputs
are rejected. Values are JSON, never pickled Python objects.

## Recovery contract

- Each model turn, validation result and tool execution is a separate Activity.
  Recorded results are replayed without repeating completed provider calls.
  Calls whose completion was lost may repeat and be billed again.
- Transient provider connection failures and selected HTTP 408/429/5xx errors
  receive at most three attempts. SDK retries are disabled in durable mode.
  Authentication failures, refusals and invalid configuration are nonretryable.
  Invalid model results consume the separate existing repair budget.
- Tool writes use fsynced intentions, atomic replacement, before/after content
  hashes and completion receipts. An external edit produces a conflict, never a
  blind overwrite. Permission revocations narrow the original permission snapshot.
- Memory operations have stable IDs and file projections reconciled from the
  journal. An interrupted append is recovered once. Memory is still captured once
  at run start and newly saved notes reach later runs.
- Arbitrary commands are never automatically executed a second time after an
  uncertain start. Temporal can retry the **receipt check**, which reports
  `needs_attention`. It does not blindly rerun the command.
- The agent's own tools (`tools=`) are treated the same way: the intent is recorded
  before the function runs and the result after, so a completed call is replayed,
  not repeated, and one interrupted mid-call waits for `resolve()`. Tools named in
  `retry_safe_tools=` (a lookup, or an action that checks whether it already
  happened) run again instead. A tool's description, arguments and retry marking
  are part of the task's fingerprint.
- Read hashes, native provider reasoning/signatures, pending calls and all repair
  counters survive restarts. Native multi-tool replies are executed in order.
- Commands support cooperative cancellation and process-group cleanup. A hard
  worker kill can leave subprocesses running. Inspect/stop those processes before
  resolving an uncertain command. Agent permissions are not an OS sandbox.

To reconcile an uncertain effect, using an operator client with access to the
Temporal namespace:

```python
status = await handle.status()
await handle.resolve(
    status["operation_id"],
    "complete",
    evidence="Checked process exit and confirmed the expected external effect",
    output="Recovered output from the operation",
)
# Alternatively: "abort", or "retry" after verifying the old process is stopped.
```

Only `complete`, `abort`, and `retry` are supported. Resending the same resolution
is idempotent. Different evidence identifies a deliberate later resolution.
For a file conflict, `retry` is refused: reconcile its expected contents or
explicitly complete/abort the operation. These operations are not model tools.
Restrict namespace access: Temporal authentication, not a string called
`evidence`, identifies the operator.

## Storage, ownership and upgrades

Use a persistent local filesystem with SQLite locking and atomic rename. The
initial adapter supports restarting on the **same volume and absolute paths**;
it does not replicate workspaces between hosts. Agent workspace roots may not
overlap and an agent memory folder cannot be shared across durable workspaces.
The worker writes `.thunc-temporal-owner` markers. Local `agent.run` refuses these
workspaces. Do not remove markers to bypass ownership while runs or subprocesses
are active. A workspace has one coordinator across task/deployment versions.
Terminated coordinators are not silently recreated.

Keep the Temporal database, worker journal, artifacts and workspace together in
your recovery plan. Temporal history does not back up your files. Missing/corrupt
artifacts fail loudly. Prompt transcripts are immutable JSON artifacts referenced
by SHA-256; a state snapshot has a 16 MiB ceiling. Submissions and final results
have a 256 KiB inline budget. Return a small application-managed artifact reference
for larger final results. There is no automatic context summarization or garbage
collection. Request-ID tombstones are retained permanently in the journal; result
retention in Temporal follows the namespace. After history expires, repeat the
original `start` request to fetch the journal's retained result.

Use explicit task versions and retain old registrations/workers until their runs
drain. Fingerprints detect changed prompts, schemas, models and nonsecret config;
they do not make incompatible code replay-safe. Bump versions when changing
validators or tool semantics. Use versioned task queues for incompatible worker
releases, while the original coordinator worker remains available. Workflow type
names and state format are versioned separately. Always replay saved histories
before changing orchestration. Changing the service endpoint/namespace of a
workspace requires an offline migration; it must not create a second owner.

`Runtime.connect` and `Worker.connect` forward client options such as `tls`,
`api_key` and `data_converter` to Temporal. A configured PayloadCodec/failure
converter protects service payloads as configured; **it does not encrypt local
SQLite files or artifact files**. Use encrypted storage and restricted filesystem
permissions for sensitive transcripts. Credentials stay in the worker environment,
not in task inputs, search attributes or artifact metadata.

## Compose tasks

`pipeline.py` demonstrates classify → agent analysis → typed summary using native
Temporal Workflows and `thunc.temporal.adapters.execute_task`. Register
`ReviewPipeline` on a separate normal Temporal worker. The helper's Activity only
submits/awaits the stable run; agent steps still execute separately. Retrying the
waiting Activity reattaches. Canceling that wait requests cancellation of the
agent. Do not invoke a task against a workspace from a run already holding that
same workspace's lane: that would wait on itself. The helper's wait has a seven-day
attempt limit; long operator pauses can outlive that wait, while the underlying
run remains inspectable.

## Tests and compatibility

```sh
pip install -e '.[anthropic,openai,dev,temporal-test]'
pytest
THUNC_TEMPORAL_TESTS=1 pytest -c pytest-temporal.ini tests/temporal
```

The integration suite downloads the official local dev service into the test
folder unless `THUNC_TEMPORAL_CLI` points at an installed binary. Set
`THUNC_TEMPORAL_DOWNLOAD_DIR` to reuse a download. It makes no model calls.
It exercises actual service history, worker termination, replay, rollover,
idempotency, cancellation, permissions and filesystem recovery boundaries.

Temporal SDK 1.34.0 is the tested baseline; the optional dependency permits
`>=1.34,<2`. Core CI covers Python 3.10–3.14; service integration CI runs 3.10 and
3.14 on Linux. Windows paths remain type-checked; do not infer a Windows durability
guarantee from that. Live provider smoke tests remain a separate opt-in task.
