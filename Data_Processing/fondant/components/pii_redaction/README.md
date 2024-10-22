# PII redaction

### Description
This component detects and redacts Personal Identifiable Information (PII) from code. 
Redaction means that sensitive data is replaced by random data.

The code is based on the PII removal code used as part of the 
[BigCode project](https://github.com/bigcode-project/bigcode-dataset/tree/main/pii).

#### PII detection

The component detects emails, IP addresses and API/SSH keys in text datasets (in particular 
datasets of source code). Regexes are used for emails and IP addresses (they are adapted from 
[BigScience PII pipeline](https://github.com/bigscience-workshop/data-preparation/tree/main/preprocessing/training/02_pii)). 
The [`detect-secrets`](https://github.com/Yelp/detect-secrets) package is used for finding 
secrets keys. Additionally filters are implemented on top to reduce the number of false 
positives, using the [gibberish-detector](https://github.com/domanchi/gibberish-detector) package.

#### PII redaction

PII is replaced by random data which is stored in the `replacements.json` file.
A component that detects and redacts Personal Identifiable Information (PII) from 
code.


### Inputs / outputs

**This component consumes:**

- code
    - content: string

**This component produces:**

- code
    - content: string

### Arguments

This component takes no arguments.

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


pii_redaction_op = ComponentOp.from_registry(
    name="pii_redaction",
    arguments={
        # Add arguments
    }
)
pipeline.add_op(pii_redaction_op, dependencies=[...])  #Add previous component as dependency
```

