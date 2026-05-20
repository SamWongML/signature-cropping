# Deploy

ECS Fargate, x86_64, 2 vCPU / 4 GB, internal ALB. See
`docs/ARCHITECTURE.md` §8 for the deployment posture.

## Files

- `ecs-task-definition.json` — task def template; replace `<ACCOUNT_ID>`,
  `<REGION>`, `<TAG>` before `aws ecs register-task-definition --cli-input-json`.

## Pre-flight

1. ECR repository `sigcrop` with image scanning enabled.
2. IAM roles: `sigcrop-task-execution` (ECR pull, logs) and `sigcrop-task`
   (S3 GetObject/PutObject scoped to the bucket prefix only).
3. Secrets Manager: `sigcrop/token-hmac-key` (the bearer-token HMAC key).
4. Private subnets only; no NAT egress at inference time.
5. Model weights committed under `models/` in the image — not pulled at runtime.
