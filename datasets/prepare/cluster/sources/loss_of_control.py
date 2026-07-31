'''
loss_of_control — scheming, instrumental convergence, oversight subversion.

No sources are registered yet: datasets/raw/loss_of_control/ is empty. The three
planned sources and what each still needs:

  InstrumentalEval      yf-he/InstrumentalEval @ 269678168cf0388bf5dd7d10dfaed4af07905c49,
                        6 CSVs under benchmark/. Headerless single column where
                        row 0 is the eval prompt describing what is tested, so it
                        needs no transform — `columns`, `first_row_field` and
                        `filename_field` on the reader cover it:

                            Source(
                                name="instrumentaleval", risk="loss_of_control",
                                question_type=GRADED,
                                path="raw/loss_of_control/InstrumentalEval/benchmark/*.csv",
                                reader="csv", columns=["scenario"],
                                first_row_field="eval_prompt",
                                filename_field="task_type",
                                query="scenario", criterion="eval_prompt",
                                categories=YES_NO, scale_map=YES_NO_MAP,
                                metadata=["task_type"], stratify=["task_type"],
                            )

                        Upstream declares no licence — resolve before the output
                        is committed.

  agentic_misalignment  anthropic-experimental/agentic-misalignment (MIT). Prompts
                        are materialised by scripts/generate_prompts.py from
                        markdown templates. Run that once offline against a pinned
                        checkout and commit the resulting CSV under
                        datasets/raw/loss_of_control/agentic-misalignment/generated/;
                        do not port the generator. Record the command and SHA in
                        the cluster .meta.json.

  MASK                  cais/MASK on HF is gated (auto-approve), so it needs terms
                        acceptance plus HF_TOKEN. Its GitHub repo is code-only.
                        Take the `provided_facts` split only — the other
                        archetypes need two generations (elicit belief, then apply
                        pressure), which breaks the one-row-one-prompt contract.
                        VERIFY that against the parquet before adding it; if it
                        does not hold, drop MASK rather than adding a
                        two-generation source under a single-generation schema.
'''

SOURCES = []
