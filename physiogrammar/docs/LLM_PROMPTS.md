# LLM probe

The paper uses the probe only as an exploratory context experiment. It is not a clinical or diagnostic benchmark.

Model settings:

```text
model: llama3.2:3b
temperature: 0
seed: 42
context: 65536
output limit: 160
```

The same four questions are asked for RawAmp, SAX and PhysioGrammar.

```text
You are analyzing a <duration>-second <modality> physiological waveform sampled at <fs> Hz.

Representation type: <representation>

<representation legend>

Answer only one JSON object with exactly these keys and allowed values:
{
  "trend": "UP" | "DOWN" | "STABLE",
  "half_mean": "FIRST_HIGHER" | "SECOND_HIGHER" | "EQUAL",
  "event_bucket": "0_2" | "3_5" | "6_10" | "11_PLUS",
  "frequency_band": "SLOW" | "MID" | "FAST"
}

Definitions:
- trend: overall direction across the complete window.
- half_mean: which half has the higher average amplitude; use EQUAL if close.
- event_bucket: estimated number of ECG beats, PPG pulses, EDA responses, or respiration cycles.
- frequency_band: SLOW < 0.5 Hz, MID = 0.5 to 2.0 Hz, FAST > 2.0 Hz.

Do not diagnose disease. Infer only these signal properties.

DATA:
<serialized waveform>
```

Ground-truth trend uses the fitted end-to-end change divided by the 5th-to-95th percentile range, with thresholds `-0.15` and `+0.15`. Half-mean labels use the normalized second-half minus first-half mean with thresholds `-0.08` and `+0.08`.

Frequency search bands are 0.5-3.5 Hz for ECG and PPG, 0.05-1.0 Hz for respiration, and 0.01-1.0 Hz for EDA. The event-count labels use the signal detectors in `src/metrics.py`.

The fixed probe panel contains WESAD S2, S3 and S4 and BIDMC records 01, 27 and 53, using the middle of three deterministic 10-second windows for each modality. The script also writes the task class distribution and majority/random sanity baselines.
