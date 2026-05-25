# H7 observations: vllm

- Stop reason: max_calls
- Total tool calls: 30 / 30
- Wall clock: 341.9s
- LLM time: 329.8s
- Finalised: False

## Tool usage
- read_file: 15
- list_validators: 8
- score_against: 6
- run_miner: 1

## Score vs active reference
- error: no final YAML (finalise never called)

## Tool-call sequence

1. run_miner({"engine": "vllm", "version_slug": "v0_7_3"}) -> ok
2. list_validators({"engine": "vllm", "class_name": "SamplingParams"}) -> ok
3. read_file({"path": "/tmp/trial_vllm_v0_7_3_venv/src/vllm/config.py", "line_start": 0, "line_end": 200}) -> ok
4. read_file({"path": "/tmp/trial_vllm_v0_7_3_venv/src/vllm/config.py", "line_start": 200, "line_end": 400}) -> ok
5. list_validators({"engine": "vllm", "class_name": "ModelConfig"}) -> ok
6. read_file({"path": "/tmp/trial_vllm_v0_7_3_venv/src/vllm/config.py", "line_start": 400, "line_end": 600}) -> ok
7. score_against({"emitted_yaml": "{\"schema_version\": \"1.0.0\", \"engine\": \"vllm\", \"engine_version\": \"v0_7_3\", \"invariants\": ) -> ok
8. read_file({"path": "/tmp/trial_vllm_v0_7_3_venv/src/vllm/config.py", "line_start": 600, "line_end": 800}) -> ok
9. list_validators({"engine": "vllm", "class_name": "SamplingParams"}) -> ok
10. read_file({"path": "/tmp/trial_vllm_v0_7_3_venv/src/vllm/config.py", "line_start": 800, "line_end": 1000}) -> ok
11. read_file({"path": "/tmp/trial_vllm_v0_7_3_venv/src/vllm/config.py", "line_start": 1000, "line_end": 1200}) -> ok
12. list_validators({"engine": "vllm", "class_name": "ModelConfig"}) -> ok
13. score_against({"emitted_yaml": "{\"schema_version\": \"1.0.0\", \"engine\": \"vllm\", \"engine_version\": \"v0_7_3\", \"invariants\": ) -> ok
14. read_file({"path": "/tmp/trial_vllm_v0_7_3_venv/src/vllm/config.py", "line_start": 1200, "line_end": 1400}) -> ok
15. read_file({"path": "/tmp/trial_vllm_v0_7_3_venv/src/vllm/config.py", "line_start": 1400, "line_end": 1600}) -> ok
16. list_validators({"engine": "vllm", "class_name": "SamplingParams"}) -> ok
17. score_against({"emitted_yaml": "{\"schema_version\": \"1.0.0\", \"engine\": \"vllm\", \"engine_version\": \"v0_7_3\", \"invariants\": ) -> ok
18. read_file({"path": "/tmp/trial_vllm_v0_7_3_venv/src/vllm/config.py", "line_start": 1600, "line_end": 1800}) -> ok
19. read_file({"path": "/tmp/trial_vllm_v0_7_3_venv/src/vllm/config.py", "line_start": 1800, "line_end": 2000}) -> ok
20. list_validators({"engine": "vllm", "class_name": "CacheConfig"}) -> ok
21. score_against({"emitted_yaml": "{\"schema_version\": \"1.0.0\", \"engine\": \"vllm\", \"engine_version\": \"v0_7_3\", \"invariants\": ) -> ok
22. read_file({"path": "/tmp/trial_vllm_v0_7_3_venv/src/vllm/config.py", "line_start": 2000, "line_end": 2200}) -> ok
23. list_validators({"engine": "vllm", "class_name": "SchedulerConfig"}) -> ok
24. score_against({"emitted_yaml": "{\"schema_version\": \"1.0.0\", \"engine\": \"vllm\", \"engine_version\": \"v0_7_3\", \"invariants\": ) -> ok
25. read_file({"path": "/tmp/trial_vllm_v0_7_3_venv/src/vllm/config.py", "line_start": 2200, "line_end": 2400}) -> ok
26. read_file({"path": "/tmp/trial_vllm_v0_7_3_venv/src/vllm/config.py", "line_start": 2400, "line_end": 2600}) -> ok
27. list_validators({"engine": "vllm", "class_name": "ModelConfig"}) -> ok
28. score_against({"emitted_yaml": "{\"schema_version\": \"1.0.0\", \"engine\": \"vllm\", \"engine_version\": \"v0_7_3\", \"invariants\": ) -> ok
29. read_file({"path": "/tmp/trial_vllm_v0_7_3_venv/src/vllm/config.py", "line_start": 2600, "line_end": 2800}) -> ok
30. read_file({"path": "/tmp/trial_vllm_v0_7_3_venv/src/vllm/config.py", "line_start": 2800, "line_end": 3000}) -> ok
