# Core scope

## Core contains mechanisms.

A deterministic Rust engine and research reference: observations, hypotheses,
evidence for and against, contradiction, temporal validity, supersession,
replay, provenance, structural relations, retrieval, bounded context packing,
explicit representations, and benchmarks with matched controls.

## Core does not contain applications.

The following do **not** belong in Core — they live in the product repository:

- GUI / Studio / dashboards;
- notebook (synthesised answers, draft-and-fill, persistent notes);
- agent runtime, delegation, reservations, orchestration;
- MCP workspace server;
- action execution against the user's machine;
- continuous watchers (filesystem, terminal, browser, process, agent);
- automatic or persistent memory products;
- persistent workspace UX and organisational context;
- prediction services and cross-workspace prior stores;
- SaaS connectors and industrial protocols;
- billing, licensing and commercial checkout;
- organisations, teams, RBAC, SSO, multi-tenancy, quotas;
- hosted services, fleet management, deployment tooling, SLAs.

## Contribution rule

> A feature being useful is not sufficient reason for inclusion in Core.

Pull requests adding GUI, notebook, orchestration, agent runtime, SaaS
integrations, persistent workspace UX, watchers, or enterprise functionality
will be rejected from Core even if technically sound.

## The promotion rule (research → Core)

An experimental mechanism moves from `research/` into the stable library only
when all of these hold:

1. the question is stated, in one sentence, in falsifiable form;
2. it has a control or null that could have killed it;
3. the benchmark can detect a known-bad alternative (self-test);
4. its result survived that comparison;
5. its limitations are written down;
6. a simpler reusable abstraction exists for it;
7. adding it needs no product infrastructure (no server, no account, no model
   download, no network).

Failing any one of these: it stays in `research/`.

## The promotion rule (Core → Product)

Value from *composition* belongs to the product. Core exposes the instruments;
the product decides when, where and how to apply them automatically.
