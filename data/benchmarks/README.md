```
# ----- Add New Datasets Here -----
# format(self) must return a CSV or JSON (preferrably CSV)
# ----- CSV/JSON Formatting Guide -----
# input	    str | list[ChatMessage]	    The input to be submitted to the model.
# choices	list[str] | None	        Optional. Multiple choice answer list.
# target	str | list[str] | None	    Optional. Ideal target output. May be a literal value or narrative text to be used by a model grader.
# id	    str | None	                Optional. Unique identifier for sample.
# metadata	dict[str | Any] | None	    Optional. Arbitrary metadata associated with the sample.
# sandbox	str | tuple[str,str]	    Optional. Sandbox environment type (or optionally a tuple with type and config file)
# files	    dict[str | str] | None	    Optional. Files that go along with the sample (copied to sandbox environments).
# setup	    str | None	                   Optional. Setup script to run for sample (executed within default sandbox environment).
```